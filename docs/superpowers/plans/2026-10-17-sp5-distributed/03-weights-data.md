# SP5 Plan — Part 3: Weights and trajectories across machines

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following blocks of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; read the spec first, it is the binding authority):
- **4.5 (ход версий):** the hub's version registry reaches every node (announcements, `Welcome`), versions never go back on a node, `GetLatest(agent, since_version)` on the learner node's data port answers without the hub (T3.1). The registry itself and generations are T2.1 / T4.1; `hub_lost_collect_sec` is T4.4.
- **4.6 (веса на машине) across machines:** one network download per (machine, agent, version) into the machine's shared-memory cache whatever the number of workers (criterion 4), `teacher_active` carried to remote workers, snapshots / frozen / teacher / `init` weights by id from the hub on any node, the evicted-snapshot hold (`checkpoints/.evicted/`), lineup after cache and `lineup_fetch_failed` counted by the hub (T3.1). The cache library is T1.5, the lineup-after-cache rule on one node T2.4.
- **4.7 across machines:** chunk routing between nodes (one stream per target node, frames forwarded undecoded), end-to-end backpressure, at-most-once delivery with every loss counted by reason, collecting seats only where a learner is reachable (T3.2); `learner.max_policy_lag` and the policy-lag percentiles (T3.3). Same-machine connections and the learner's single intake path are T2.2.
- **4.8 (пересылка через хаб):** `RelaySend` / `RelayInbox`, bounded buffers per node pair with credit-based flow control, relayed latest weights (one upload per version), `relay_bytes`, the local relay to the hub machine's learners, `distributed.relay: false`, the path choice by `connect_timeout_sec` (T3.2).

Not in this part: learner placement, generations being real and fencing (T4.1 — until then every learner is generation 1, T2.5), recovery points (T4.2), hub-away behaviour beyond `GetLatest` (T4.4: lineups past their TTL, `hub_lost_collect_sec`, result buffering), evaluators (T5.1), the `system` record keys and the «Nodes» section (T7.1: this part makes the numbers exist and persist; T7.1 prints them), `--ban` (T6.3: this part only checks a ban list on the data port).

**What the part delivers.** After T3.3:
- a worker on any node plays with the newest weights of a learner on any other node: the learner's node announces each version to the hub, the hub announces it to every node, each node downloads it once (directly from the learner node's data port, or through the hub when that port is unreachable) and publishes it into its shared-memory cache; without the hub, nodes ask the learner's data port directly;
- snapshot, frozen, teacher and `init` weights reach every node by id; a snapshot evicted while a granted lineup names it stays downloadable from `checkpoints/.evicted/` for at most `2 × match_refresh_interval_sec`; a failed download keeps the old lineup and is counted (`lineup_fetch_failed` in the hub);
- a worker on any node collects for a learner on any other node: chunk frames go to the local node, then in one stream per target node to that node's data port (mutual TLS) or through the hub relay, and from the receiving node to the learner's chunk connection; nothing in between decodes them; a slow learner slows the workers on every machine; every lost frame is counted by reason (`disconnect`, `no_learner`, `stale`, `stale_generation`, `invalid`, `overflow`);
- `learner.max_policy_lag` drops too-old chunks (`stale`), and every `train` record carries `policy_lag_p50` / `policy_lag_p95` / `policy_lag_max` and `dropped_stale_chunks`;
- the hub keeps monotonic run totals of node-side and relay counters (`CounterBook`) in `hub_state.json`.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract, cross-part notes). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (owner's machine: ROS sets `PYTHONPATH`).
- "Full fast suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp5_` and have unique basenames; support modules (`cluster_kit`, `fault_proxy`) are imported by bare name and added to `known-first-party` in `pyproject.toml` by the task that creates them.
- Files only under `tmp_path` (the autouse `_isolate_colosseum_dirs` of T1.3 redirects `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `COLOSSEUM_SHM_DIR`; two nodes of one test get two identity roots through `NodeOptions.identity_root`).
- At most **2 loaded CPU processes per fast test** (workers, learners; hub and node agents running in threads of the test process are not counted), at most 3 per slow test. A test longer than ~20 s is `@pytest.mark.slow`.
- Durations and cadences use `time.monotonic()`; waits are on counters (`cluster_kit.wait_until`), never on wall-clock values. Messages go through `logging`, never `warnings.warn`.
- gRPC servers in tests bind port 0 (`DataServer.start(0, ...)`, `HubOptions(port=0)`); never a bind-then-release free-port helper.
- Commits use conventional prefixes and carry no attribution lines; push after every task (`git push`, branch `sp5-distributed` or the worktree branch the controller names).

**Cross-task notes.**
- **Order:** T3.1 → (T3.2 ‖ T3.3). T3.1 creates the shared pieces both later tasks use: `colosseum.metrics.counters.CounterBook` (monotonic run totals), `colosseum.node.dataport` (`DataServer`, `probe_direct`, `fetch_latest`), `Heartbeat.counters` and the hub's book, `tests/cluster_kit.py`, `tests/fault_proxy.py`.
- **Parallel pair T3.2 ‖ T3.3 and their files** (overview index): T3.2 owns `node/router.py` (new), `hub/relay.py` (new), `node/dataport.py` (the `Chunks` method only), `hub/server.py` (relay servicers, `Download(kind="latest")`, `Upload(kind="relay_latest")` only), `node/agent.py` (routing section), `hub/hub.py` (the collect rule and the relay counters only), `hub/state.py` (`collect_allowed`, `strip_collect`); T3.3 owns `learner/learner.py`, `net/local.py` (the intakes' lag rule), `node/learner_main.py` (one keyword), `metrics/hub.py`, `core/config.py` (one `LearnerConfig` validator). No file is edited by both.
- **Interim rules this part lifts** (overview, "Node rules that cross tasks"): "until T3.1 latest weights exist only on the learner's machine" (T3.1 lifts it); "until T3.2 the hub grants collecting seats only for agents whose learner runs on the requesting node" (T3.2 lifts it).
- **Learner placement before T4.1:** in `colosseum hub` mode the hub of T2.5 assigns every trainable agent's learner to a node with a learner slot (assumption A3 below). Cluster tests start the learner node first and wait until its learner announced a version before they start the other node.
- **Threads:** the node's new threads (fetch jobs in the transfer pool, `TargetSender`, `RouteReader`, `RelayReceiver`, data-port servicers) never touch main-loop state: they post into the node inbox or use their own thread-safe objects (`RouteTable`, `LocalDelivery`, `Counters`). The hub's relay state (`RelayHub`, `LatestRelay`) is used by servicer threads; the main loop only publishes into it (`set_online`, `publish_versions`) and reads `stats()`.
- **Counters:** every counter of this part is cumulative per process (node process, hub process, learner incarnation) and is summed into run totals by a `CounterBook` keyed by its source, so totals stay monotonic across joins, restarts and `--resume`. Names: `lineup_fetch_failed`, `snapshot_holds`, `weight_fetches`, `weight_fetch_bytes`, `weight_fetch_failed`, `weight_fetch_not_modified`, `weight_fetch_redundant` (T3.1); `relay_bytes`, `relay_frames`, `chunks_sent.direct`, `chunks_sent.relay`, `chunks_delivered`, `dropped_chunks.<reason>` (T3.2, T3.3). T7.1 turns them into `system` keys.

**Verified while writing this part.** Every new library module of this part was written in a scratch package (`/tmp/.../scratchpad/part03/overlay/`) against the real repo (`colosseum` from `src/`, grpcio 1.84, cryptography 50, Python 3.12) with stand-ins for the T1.x/T2.x modules it imports (a protobuf subset of the contract messages generated with `grpc_tools`, `send_parts` / `receive_parts`, the shm cache pointer files, `peek_meta`), and its tests were run there: `CounterBook`, `FetchPlanner`, `SnapshotHolds` + `snapshot_refs`, `RelayHub`, `LatestRelay`, `TcpFaultProxy`, `ChunkRouter` (without gRPC), `DataServer` + `fetch_latest` + `probe_direct` over real mutual TLS (a P-256 CA and node certificates made with `cryptography`), and the `CheckpointManager` hold (patched onto the real class) — 46 tests, green repeatedly (five and three runs), `ruff` clean on the modules. An end-to-end scratch run pushed 600–900 frames of 20 KB from a `TargetSender` to a slow consumer on a `multiprocessing` pipe: direct (learner-paced: 1.6 s for 600 frames at 2 ms each, no loss), relay only (credit flow through `RelayHub`, no loss), a 1.5 s `TcpFaultProxy.cut` on the direct path (switch to relay; delivered + counted = 905 ≥ 900 produced: the broken stream's frames are an upper bound of the loss) and the same cut with the relay off (direct → down → direct; delivered + counted ≥ produced). Findings built into the tasks: (1) **grpc-python servers set `SO_REUSEPORT` by default** — a second server binds the same port and the kernel splits connections between them; the data port's "next free port" rule needs `("grpc.so_reuseport", 0)` (also a requirement on T1.3's `server_options`, contract note 15); (2) `grpc.channel_ready_future` returns a `grpc.Future`, not a `concurrent.futures.Future` (`futures.wait` fails on it); (3) a `threading.Thread` subclass must not name an attribute `_stop` (it shadows `Thread._stop` and `join` fails); (4) gRPC flow control on a client stream holds about 13 frames of 260 KB (~3.4 MB) in flight to a slow server — so a cut loses at most that plus one rotated stream; (5) the mTLS peer's node id is `context.auth_context()["x509_common_name"][0]`; refusals of a certificate from another CA or of no certificate surface as `UNAVAILABLE` (handshake failure) on the client. Code that edits T2.x modules (the hub, the node agent, the servers) could not be run: those modules do not exist yet; the steps give the behaviour, the edit points and the tests.

**Traceability (spec item → task / test).**

| Spec | Item | Task | Tests |
|---|---|---|---|
| 4.5 | hub announces every version to every node; `Welcome` carries the registry | T3.1 | `test_sp5_weights_cluster.py::test_a_second_node_plays_with_the_learner_nodes_weights_and_the_frozen_blob`, `test_sp5_node_fetch.py::test_one_download_per_version_whatever_the_number_of_workers` |
| 4.5 | versions never go back on a node | T3.1 | `test_sp5_fetch_planner.py::test_announcements_never_go_back_but_a_new_address_replaces_the_source`, `::test_a_version_at_or_below_the_cached_one_is_redundant` |
| 4.5 | `GetLatest` without the hub | T3.1 | `test_sp5_data_port.py::test_get_latest_streams_the_cached_frame_and_answers_not_modified`, `test_sp5_node_fetch.py::test_without_the_hub_versions_come_from_get_latest` |
| 4.6 | one network download per (machine, agent, version) (criterion 4) | T3.1 | `test_sp5_node_fetch.py::test_one_download_per_version_whatever_the_number_of_workers`, `test_sp5_weights_two_nodes_slow.py` |
| 4.6 | `teacher_active` reaches remote workers | T3.1 | `test_sp5_node_fetch.py::test_teacher_active_off_reaches_the_remote_cache` |
| 4.6 | frames carry agent, generation, version, `teacher_active`; checked on arrival | T3.1 | `test_sp5_data_port.py::test_a_fetched_frame_is_checked_against_its_metadata` |
| 4.6 | snapshots / frozen / teacher / `init` by id on any node, architecture with them | T3.1 | `test_sp5_weights_cluster.py::test_a_second_node_plays_with_the_learner_nodes_weights_and_the_frozen_blob`, `::test_teacher_and_init_blobs_reach_a_second_node` |
| 4.6 | evicted snapshot dropped from a node's cache when unreferenced | T3.1 (T2.4's rule, checked across machines) | `test_sp5_snapshot_hold.py::test_snapshots_evicted_while_referenced_stay_downloadable` |
| 4.2 | evicted-snapshot hold (`checkpoints/.evicted/`, ≤ 2 × refresh) | T3.1 | `test_sp5_snapshot_holds.py`, `test_sp5_checkpoint_hold.py`, `test_sp5_blob_store_hold.py`, `test_sp5_snapshot_hold.py` |
| 4.2 | failed download keeps the old lineup, `lineup_fetch_failed`, re-request | T3.1 | `test_sp5_node_fetch.py::test_a_failed_first_fetch_holds_the_lineup_and_counts_lineup_fetch_failed`, `test_sp5_blob_store_hold.py::test_a_released_snapshot_is_not_found` |
| 4.6 | `/dev/shm` full → ERROR with the hint | T3.1 | `test_sp5_node_fetch.py::test_a_full_cache_is_an_error_with_the_hint_and_counted` |
| 3.1 | node data port on every interface, next free port, announced addresses, `port != data_port` | T3.1 | `test_sp5_data_port.py::test_a_taken_data_port_moves_to_the_next_free_one`, `::test_data_addresses_start_with_loopback_for_a_loopback_hub`, `test_sp5_version_source.py`, `test_sp5_weights_cluster.py` |
| 3.2 | mutual TLS on data ports, fixed name `colosseum-node`, CA + not banned | T3.1 | `test_sp5_data_port.py::test_peers_without_a_certificate_of_the_ca_or_banned_are_refused` |
| 3.3 | `--insecure` data port still needs the token | T3.1 | `test_sp5_data_port.py::test_insecure_data_port_needs_the_token_secret` |
| 4.7 | chunk routing, one stream per target node, frames undecoded | T3.2 | `test_sp5_chunk_router.py::test_frames_go_to_their_agents_target_and_unknown_agents_are_counted`, `test_sp5_chunks_cluster.py::test_a_worker_on_another_node_trains_the_learner_directly` |
| 4.7 | end-to-end backpressure | T3.2 | `test_sp5_chunk_router.py::test_a_slow_target_blocks_the_worker_instead_of_dropping`, `test_sp5_data_port_chunks.py::test_a_slow_learner_backpressures_the_sending_node`, `test_sp5_relay_hub.py::test_a_full_pair_buffer_blocks_the_sender_until_the_receiver_grants_credit` |
| 4.7 | at-most-once, every loss counted by reason | T3.2, T3.3 | `test_sp5_chunk_router.py`, `test_sp5_data_port_chunks.py::test_frames_without_a_learner_or_with_a_dead_learner_are_counted`, `test_sp5_chunks_cluster_slow.py::test_a_cut_is_survived_and_every_chunk_is_counted`, `test_sp5_policy_lag.py` |
| 4.7 | collecting seats only for agents with a reachable running learner | T3.2 | `test_sp5_collect_rule.py`, `test_sp5_chunks_cluster.py::test_relay_off_keeps_an_unreachable_node_from_collecting` |
| 4.7 | `learner.max_policy_lag` (`stale`) | T3.3 | `test_sp5_policy_lag.py::test_max_policy_lag_drops_old_chunks_and_counts_them_stale`, `::test_connection_intake_drops_stale_frames_before_decoding`, `test_sp5_policy_lag_run.py` |
| 4.7 | lag p50 / p95 / max in `train` always | T3.3 | `test_sp5_policy_lag.py::test_lag_metrics_are_percentiles_of_the_batch`, `::test_the_learner_reports_lag_percentiles_and_keeps_the_old_keys` |
| 4.8 | relay when direct fails within `connect_timeout_sec` | T3.2 | `test_sp5_chunks_cluster.py::test_relay_carries_chunks_and_weights_when_the_data_port_is_unreachable` |
| 4.8 | bounded pair buffers, flow control, `overflow` | T3.2 | `test_sp5_relay_hub.py` |
| 4.8 | relayed weights in parts, one upload per version | T3.2 | `test_sp5_relay_hub.py::test_latest_relay_pulls_once_and_serves_every_waiter_from_one_upload`, `test_sp5_relay_servicers.py::test_relayed_latest_weights_are_uploaded_once_for_many_downloads` |
| 4.8 | local relay to the hub machine's learners | T3.2 | `test_sp5_chunks_cluster.py::test_relay_carries_chunks_and_weights_when_the_data_port_is_unreachable` (a learner node without a reachable data port is the same path), `test_sp5_chunks_cluster_slow.py::test_a_relay_only_learner_survives_a_cut_of_the_hub_relay_stream` |
| 4.8 | path visible (WARNING once per pair, `Heartbeat.paths`), `relay_bytes` | T3.2 | `test_sp5_chunks_cluster.py::test_relay_carries_chunks_and_weights_when_the_data_port_is_unreachable` |
| 4.8 | `distributed.relay: false` | T3.2 | `test_sp5_relay_servicers.py::test_relay_off_refuses_relay_streams`, `test_sp5_chunks_cluster.py::test_relay_off_keeps_an_unreachable_node_from_collecting` |
| 8 | monotonic counters across sessions and resume | T3.1 | `test_sp5_counter_book.py` |
| §6 | TCP-proxy fault injection helper | T3.1 (used by T3.2) | `test_sp5_fault_proxy.py` |

---

### Task T3.1: Weights across machines: announcements, the data port and `GetLatest`, one fetch per machine, blobs by id with the evicted-snapshot hold, `lineup_fetch_failed`

Spec 4.5, 4.6, 4.2 (lineup after cache across machines, the evicted-snapshot hold, `lineup_fetch_failed`), 3.1 (the node data port), 3.2 / 3.3 (data-port security), 8 (monotonic counters). After T2.5 the hub keeps a `VersionRegistry`, a node's own learner publishes into its machine's cache, and a node fetches blobs (snapshots, frozen, teacher, `init`) from the hub into its cache before it passes a lineup to a worker (T2.4). This task makes latest weights of a learner on another machine reach a node: the hub fills each accepted `VersionInfo` with the announcing node's data addresses and announces it to every node; every node runs a data port (`NodeData`, mutual TLS) whose `GetLatest` serves its learners' newest frame from the cache; a `FetchPlanner` on every node decides which agents to fetch (one transfer per agent in flight, the newest version only, at most once per `rollout.weight_sync_interval_sec`, first weights at once); the transfer pool downloads, the node checks the frame (digest, header) and publishes it into its cache (one download per version, criterion 4); without the hub the planner polls `GetLatest` directly. Then (split part b) the hub holds evicted snapshots that a granted lineup still names (`checkpoints/.evicted/`, ≤ `EVICTED_HOLD_FACTOR × match_refresh_interval_sec`), and node counters (`lineup_fetch_failed`, fetch counts) reach the hub in `Heartbeat.counters` and are summed by a `CounterBook` into `hub_state.json`. The relay path for weights (a learner node whose data port is unreachable) is T3.2.

**Split point:** dispatch (a) Steps 1–9 (counters, planner, data port, proto additions, node fetcher, hub fan-out and counters, `cluster_kit`, the fetch tests), then (b) Steps 10–16 (the hold: `SnapshotHolds`, `CheckpointManager`, `BlobStore`, hub wiring, the hold tests, the slow two-node test). Each dispatch ends with the full fast suite, a commit and a push.

**Files:**
- Create: `src/colosseum/metrics/counters.py` (`CounterBook`)
- Create: `src/colosseum/node/fetch.py` (`FetchPlanner`, `FetchJob`, `FetchStats`)
- Create: `src/colosseum/node/dataport.py` (`DataServer`, `probe_direct`, `fetch_latest`, `check_fetched`, `FetchError`, `FetchedLatest`, `data_addresses`, `frame_digest`, `DATA_PORT_TRIES`, `DATA_SERVER_THREADS`)
- Create: `tests/cluster_kit.py`, `tests/fault_proxy.py`
- Modify: `src/colosseum/net/proto/colosseum_v1.proto` + regenerated `colosseum_v1_pb2.py` / `_pb2_grpc.py` (`scripts/gen_proto.sh`): `Heartbeat.counters`, `NodeMessage.addresses` (`DataAddresses`), `Welcome.banned_node_ids`, `Announce.banned_node_ids`
- Modify: `src/colosseum/node/weights.py` (`WeightCacheReader.open_latest`)
- Modify: `src/colosseum/node/agent.py` (`NodeOptions.data_addresses`, the data port, the fetch section, `Heartbeat.counters`, `NodeAgent.counters()`, `NodeAgent.fetch_history()`)
- Modify: `src/colosseum/hub/state.py` (`version_source`, `SnapshotHolds`, `snapshot_refs`)
- Modify: `src/colosseum/hub/hub.py` (version fan-out, `_on_data_addresses`, the counter book in `_on_heartbeat` and `hub_state.json`, the hold wiring)
- Modify: `src/colosseum/hub/server.py` (`NodeMessage.addresses` forwarded as `NodeMessageIn`; `BlobStore` falls back to `checkpoints/.evicted/`, `BlobStore.forget`)
- Modify: `src/colosseum/coordinator/checkpoint_manager.py` (`EVICTED_DIR`, `set_eviction_hold`, `held_dir`, `release_held`, `clear_held`)
- Modify: `pyproject.toml` (`cluster_kit`, `fault_proxy` in `known-first-party`)
- Test: `tests/unit/test_sp5_counter_book.py`, `tests/unit/test_sp5_fetch_planner.py`, `tests/unit/test_sp5_data_port.py`, `tests/unit/test_sp5_fault_proxy.py`, `tests/unit/test_sp5_version_source.py`, `tests/unit/test_sp5_snapshot_holds.py`, `tests/unit/test_sp5_checkpoint_hold.py`, `tests/unit/test_sp5_blob_store_hold.py`, `tests/integration/test_sp5_node_fetch.py`, `tests/integration/test_sp5_weights_cluster.py`, `tests/integration/test_sp5_snapshot_hold.py`, `tests/integration/test_sp5_weights_two_nodes_slow.py`

**Interfaces:**
- Consumes: `VersionRegistry`, `VersionInfo`, `NodeEntry`, `CellKey`, `EVICTED_HOLD_FACTOR`, `HubStateData` (T2.1); `WeightCacheWriter` / `WeightCacheReader` / `CacheKey` / `CacheFullError` (T1.5); `encode_weights` / `WeightsMeta` (T1.1), `peek_meta`, `payload_byte_cap`, `FrameError` (T1.1); `send_parts` / `receive_parts` / `PART_BYTES` / `channel_options` / `server_options` (T1.3, T2.3); `NodeIdentity`, `data_server_credentials`, `data_channel_credentials`, `NODE_SERVER_NAME`, `MD_TOKEN`, `HubIdentity` (T1.3); `BlobStore`, `build_server`, `NodeMessageIn` (T2.3); `HubSession` (T2.3); `NodeAgent`, `NodeOptions`, the node inbox and transfer pool, the lineup-after-cache machinery (T2.4); `Hub`, `HubOptions`, `_on_version`, `_on_heartbeat`, `_on_lineup_request`, `_on_snapshot_evicted`, `read_hub_json` (T2.5); `colosseum.arena.entities.snapshot_version`.
- Produces (contract additions, see Contract notes): `CounterBook` (`report`, `add`, `get`, `totals`, `grouped`, `state`, `from_state`); `FetchPlanner` / `FetchJob` / `FetchStats`; the `dataport` names above; `WeightCacheReader.open_latest(agent_id) -> tuple[BinaryIO, dict] | None`; `version_source(info, entry) -> VersionInfo`; `SnapshotHolds`, `snapshot_refs`; `CheckpointManager.{set_eviction_hold, held_dir, release_held, clear_held}`, `EVICTED_DIR`; `BlobStore.forget(kind, agent_id, ident)`; `NodeOptions.data_addresses: tuple[str, ...] | None = None`; `NodeAgent.counters() -> dict[str, int]`, `NodeAgent.fetch_history() -> dict[str, list[int]]`; proto fields `Heartbeat.counters`, `DataAddresses`, `banned_node_ids`; `tests/cluster_kit.py` (`cluster_config`, `HubThread`, `start_hub`, `NodeThread`, `start_node`, `cluster`, `wait_until`, `read_hub_state`, `journal_results_by_node`, `hub_version`).

#### Part (a)

- [ ] **Step 1: Write the failing pure tests (counters, planner)**

Create `tests/unit/test_sp5_counter_book.py`:
```python
"""CounterBook (SP5 T3.1, spec block 8): run totals of cumulative per-source counters stay monotonic across
sources that join, restart and leave, and a resumed book counts nothing twice."""
from __future__ import annotations

from colosseum.metrics.counters import CounterBook


def test_totals_add_the_increase_of_every_source():
    book = CounterBook()
    assert book.report("n1/s1", {"lineup_fetch_failed": 2, "weight_fetches": 5}) == {
        "lineup_fetch_failed": 2, "weight_fetches": 5}
    assert book.report("n1/s1", {"lineup_fetch_failed": 2, "weight_fetches": 7}) == {"weight_fetches": 2}
    book.report("n2/s1", {"weight_fetches": 1})
    assert book.totals() == {"lineup_fetch_failed": 2, "weight_fetches": 8}


def test_a_new_session_of_a_node_adds_to_the_totals_and_never_lowers_them():
    book = CounterBook()
    book.report("n1/s1", {"dropped_chunks.disconnect": 4})
    book.report("n1/s2", {"dropped_chunks.disconnect": 1})      # the node process restarted: a new session
    assert book.get("dropped_chunks.disconnect") == 5
    book.report("n1/s1", {"dropped_chunks.disconnect": 3})      # a late, lower report of the old session
    assert book.get("dropped_chunks.disconnect") == 8           # counted as a restart under the same key
    assert book.get("missing") == 0


def test_malformed_values_are_ignored():
    book = CounterBook()
    book.report("n1/s1", {"a": -3, "b": 1.5, "c": True, "d": "7", "e": 2})
    assert book.totals() == {"e": 2}


def test_a_resumed_book_continues_without_double_counting():
    book = CounterBook()
    book.report("n1/s1", {"relay_bytes": 100})
    book.add("dropped_chunks.overflow", 3)
    saved = book.state()
    book.report("n1/s1", {"relay_bytes": 150})                  # counted after the save, lost with the crash
    resumed = CounterBook.from_state(saved)
    assert resumed.totals() == {"relay_bytes": 100, "dropped_chunks.overflow": 3}
    resumed.report("n1/s1", {"relay_bytes": 180})               # the same session reports again after --resume
    assert resumed.get("relay_bytes") == 180
    assert CounterBook.from_state(None).totals() == {}


def test_grouped_nests_dotted_names_and_lists_every_member_of_a_group():
    book = CounterBook()
    book.add("relay_bytes", 10)
    book.add("dropped_chunks.stale", 2)
    reasons = ("disconnect", "no_learner", "stale", "stale_generation", "invalid", "overflow")
    assert book.grouped(groups={"dropped_chunks": reasons}) == {
        "relay_bytes": 10,
        "dropped_chunks": {"disconnect": 0, "no_learner": 0, "stale": 2, "stale_generation": 0, "invalid": 0,
                           "overflow": 0},
    }
```

Create `tests/unit/test_sp5_fetch_planner.py`:
```python
"""FetchPlanner (SP5 T3.1, spec 4.5, 4.6, criterion 4): which latest weights a node fetches from other machines and
when. One transfer per agent in flight, one download per new version, the sync cadence, first weights at once,
GetLatest polls only without the hub and only to direct sources, never the node's own learners."""
from __future__ import annotations

import pytest

from colosseum.hub.state import VersionInfo
from colosseum.node.fetch import FetchPlanner


def _info(agent="main", version=1, *, node=2, generation=1, addresses=("10.0.0.2:47102",), relay_only=False):
    return VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d", node=node,
                       addresses=addresses, relay_only=relay_only)


def _planner(**kw):
    return FetchPlanner(node=1, sync_interval_sec=kw.pop("interval", 5.0))


def test_first_weights_are_fetched_at_once_and_once():
    p = _planner()
    p.on_version(_info(version=3))
    jobs = p.due({}, hub_online=True, now=0.0)
    assert [(j.agent_id, j.since_version, j.mode, j.source.version) for j in jobs] == [("main", -1, "announced", 3)]
    assert p.in_flight("main")
    assert p.due({}, hub_online=True, now=0.1) == []             # one transfer per agent in flight
    assert p.finished("main", 3, cached=-1) is True
    assert p.stats.history == {"main": [3]} and p.stats.fetched == 1


def test_a_newer_version_is_fetched_once_per_sync_interval_while_referenced():
    p = _planner(interval=5.0)
    p.set_referenced({"main"})
    p.on_version(_info(version=1))
    [job] = p.due({}, hub_online=True, now=0.0)
    p.finished("main", 1, cached=-1)
    for v in (2, 3, 4):
        p.on_version(_info(version=v))
    assert p.due({"main": 1}, hub_online=True, now=4.9) == []    # cadence: one check per interval
    [job] = p.due({"main": 1}, hub_online=True, now=5.0)
    assert job.source.version == 4 and job.since_version == 1    # only the newest is fetched
    p.finished("main", 4, cached=1)
    assert p.due({"main": 4}, hub_online=True, now=20.0) == []   # nothing newer announced
    assert p.stats.history == {"main": [1, 4]}


def test_an_unreferenced_agent_is_fetched_only_until_something_is_cached():
    p = _planner()
    p.on_version(_info(version=1))
    [_] = p.due({}, hub_online=True, now=0.0)
    p.finished("main", 1, cached=-1)
    p.on_version(_info(version=9))
    assert p.due({"main": 1}, hub_online=True, now=100.0) == []
    p.set_referenced({"main"})
    assert [j.source.version for j in p.due({"main": 1}, hub_online=True, now=100.0)] == [9]


def test_own_learners_are_never_fetched():
    p = _planner()
    p.set_referenced({"main", "other"})
    p.on_version(_info("main", 5, node=1))                       # the learner of main runs on this node (1)
    p.on_version(_info("other", 5, node=3))
    p.set_local_agents({"other"})
    assert p.due({}, hub_online=True, now=0.0) == []


def test_a_failure_waits_one_interval():
    p = _planner(interval=2.0)
    p.on_version(_info(version=1))
    [_] = p.due({}, hub_online=True, now=0.0)
    p.failed("main")
    assert p.stats.failed == 1 and not p.in_flight("main")
    assert p.due({}, hub_online=True, now=1.9) == []
    assert len(p.due({}, hub_online=True, now=2.0)) == 1


def test_without_the_hub_direct_sources_are_polled_and_relay_only_ones_are_not():
    p = _planner(interval=1.0)
    p.set_referenced({"main", "far"})
    p.on_version(_info("main", 4))
    p.on_version(_info("far", 4, node=3, addresses=(), relay_only=True))
    jobs = p.due({"main": 4, "far": 4}, hub_online=False, now=0.0)
    assert [(j.agent_id, j.mode, j.since_version) for j in jobs] == [("main", "poll", 4)]
    assert p.finished("main", None, cached=4) is False           # GetLatest: not modified
    assert p.stats.not_modified == 1
    assert [j.agent_id for j in p.due({"main": 4}, hub_online=False, now=1.0)] == ["main"]
    assert p.finished("main", 6, cached=4) is True               # newer than announced: taken


def test_a_version_at_or_below_the_cached_one_is_redundant():
    p = _planner()
    p.on_version(_info(version=3))
    [_] = p.due({}, hub_online=True, now=0.0)
    assert p.finished("main", 3, cached=3) is False
    assert p.stats.redundant == 1 and p.stats.fetched == 0
    assert p.stats.as_counters() == {"weight_fetches": 0, "weight_fetch_failed": 0,
                                     "weight_fetch_not_modified": 0, "weight_fetch_redundant": 1}


def test_announcements_never_go_back_but_a_new_address_replaces_the_source():
    p = _planner()
    assert p.on_version(_info(version=5)) is True
    assert p.on_version(_info(version=4)) is False
    assert p.on_version(_info(version=5)) is False                # the same version from the same place
    assert p.on_version(_info(version=5, addresses=("10.0.0.9:47102",))) is True
    assert p.known("main").addresses == ("10.0.0.9:47102",)
    p.forget("main")
    assert p.known("main") is None


def test_the_interval_must_be_positive():
    with pytest.raises(ValueError):
        FetchPlanner(node=1, sync_interval_sec=0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_counter_book.py tests/unit/test_sp5_fetch_planner.py -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'colosseum.metrics.counters'` and `'colosseum.node.fetch'`.

- [ ] **Step 3: Write `src/colosseum/metrics/counters.py` and `src/colosseum/node/fetch.py`**

`src/colosseum/metrics/counters.py`:
```python
"""Run-level sums of cumulative counters reported by restartable sources (SP5 spec block 8).

A *source* is anything that counts from zero when it starts and reports its cumulative values: a node session
(``"<node_id>/<session_id>"``), a learner incarnation (``"learner/<agent>/<generation>"``), the hub process itself
(``"hub/<hub incarnation>"``). The book keeps the highest value per (source, name) and adds every increase to the
run total, so the totals never decrease while sources join, restart and leave, and a resumed hub that hears the
same sources report again (their cumulative values continue) counts nothing twice: ``state()`` saves the totals
together with the per-source values they were computed from (the ``BudgetCounter`` rule of spec 4.4, for any
number of names).

A value below the source's previous one means the source restarted under the same key (never by construction:
session ids and generations are new per start): the new value is counted from zero. Names are flat strings;
dotted names (``"dropped_chunks.disconnect"``) are grouped by ``grouped()`` for the ``system`` record and
``hub_state.json``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class CounterBook:
    """Monotonic run totals of cumulative per-source counters (module docstring). Not thread-safe: one owner
    (the hub main loop, or ``MetricsHub``)."""

    def __init__(self, totals: Mapping[str, int] | None = None,
                 sources: Mapping[str, Mapping[str, int]] | None = None) -> None:
        self._totals: dict[str, int] = {str(k): int(v) for k, v in (totals or {}).items()}
        self._sources: dict[str, dict[str, int]] = {str(s): {str(k): int(v) for k, v in vals.items()}
                                                    for s, vals in (sources or {}).items()}

    def report(self, source: str, values: Mapping[str, int]) -> dict[str, int]:
        """Take ``source``'s cumulative ``values``; return the increments applied (names with a zero increment
        are left out). Negative or non-integer values are ignored (a malformed report never lowers a total)."""
        last = self._sources.setdefault(str(source), {})
        applied: dict[str, int] = {}
        for name, raw in values.items():
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                continue
            name = str(name)
            prev = last.get(name, 0)
            delta = raw - prev if raw >= prev else raw      # a lower value: the source started over
            last[name] = raw
            if delta:
                self._totals[name] = self._totals.get(name, 0) + delta
                applied[name] = delta
        return applied

    def add(self, name: str, delta: int) -> None:
        """Count ``delta`` (>= 0) directly (an event of the book's owner, e.g. a snapshot hold of the hub)."""
        if delta > 0:
            self._totals[str(name)] = self._totals.get(str(name), 0) + int(delta)

    def get(self, name: str) -> int:
        return self._totals.get(str(name), 0)

    def totals(self) -> dict[str, int]:
        return dict(self._totals)

    def grouped(self, *, groups: Mapping[str, tuple[str, ...]] | None = None) -> dict[str, Any]:
        """Totals with dotted names nested one level: ``{"relay_bytes": 5, "dropped_chunks": {"disconnect": 2}}``.
        ``groups`` lists the members every group must show (zeros included), e.g. the six chunk-drop reasons."""
        out: dict[str, Any] = {}
        for group, members in (groups or {}).items():
            out[group] = {m: 0 for m in members}
        for name, value in sorted(self._totals.items()):
            head, dot, tail = name.partition(".")
            if dot:
                out.setdefault(head, {})
                out[head][tail] = value
            else:
                out[name] = value
        return out

    def state(self) -> dict[str, Any]:
        return {"totals": dict(self._totals), "sources": {s: dict(v) for s, v in self._sources.items()}}

    @classmethod
    def from_state(cls, data: Mapping[str, Any] | None) -> CounterBook:
        data = data or {}
        return cls(data.get("totals") or {}, data.get("sources") or {})
```

`src/colosseum/node/fetch.py`:
```python
"""Which latest weights a node fetches from other machines, and when (SP5 spec 4.5, 4.6; pure, node clock).

Weights reach a node's workers only through the machine's shared-memory cache (R10). For an agent whose learner
runs on THIS node the learner writes the cache itself; every other agent's latest weights are fetched by the node,
once per (agent, version), and published into the cache, so the number of network downloads does not depend on
the number of workers (criterion 4).

Rules (``FetchPlanner.due``):
- the planner knows, per agent, the newest ``VersionInfo`` it heard of (``Welcome.versions``, ``Announce.versions``;
  a re-announce of the same version with new addresses replaces the source);
- agents whose learner runs on this node (``set_local_agents``) are never fetched;
- at most one transfer per agent is in flight;
- with the hub online: an agent is fetched when its announced version is newer than the cached one and either
  nothing is cached yet (first weights: at once) or a latest seat of a live or pending lineup of this node names it
  (``set_referenced``); the version check runs at most once per ``rollout.weight_sync_interval_sec`` per agent
  (the cadence starts when a transfer starts, so a failure also waits one interval);
- without the hub (session down): the node asks the learner's data port directly (``GetLatest(agent,
  since_version)``) once per interval for every referenced or never-cached agent whose source has data addresses;
  a learner reachable only through the relay gets no polls (spec 4.5: weights of such a learner need the hub);
- a result never lowers the cached version (versions only grow, R9): a version at or below the cached one is
  recorded as redundant (counted; criterion 4 expects zero).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol


class VersionLike(Protocol):        # colosseum.hub.state.VersionInfo
    agent_id: str
    generation: int
    version: int
    node: int
    addresses: tuple[str, ...]
    relay_only: bool


@dataclass(frozen=True)
class FetchJob:
    """One transfer to start: ``mode`` "announced" (the hub said ``source.version`` exists) or "poll" (no hub:
    ``GetLatest`` answers "not modified" or with whatever is newer than ``since_version``)."""

    agent_id: str
    since_version: int                 # the cached version, -1 when nothing is cached
    source: VersionLike
    mode: Literal["announced", "poll"]


@dataclass
class FetchStats:
    jobs: int = 0                      # transfers started
    fetched: int = 0                   # transfers that brought a newer version (one network download each)
    not_modified: int = 0              # polls answered "not modified"
    failed: int = 0
    redundant: int = 0                 # a version at or below the cached one arrived (criterion 4: 0)
    history: dict[str, list[int]] = field(default_factory=dict)   # versions fetched per agent, in order

    def as_counters(self) -> dict[str, int]:
        """Cumulative values for ``Heartbeat.counters`` (names of the hub's CounterBook)."""
        return {"weight_fetches": self.fetched, "weight_fetch_failed": self.failed,
                "weight_fetch_not_modified": self.not_modified, "weight_fetch_redundant": self.redundant}


class FetchPlanner:
    """Fetch decisions of one node (module docstring). Not thread-safe: the node main loop owns it."""

    def __init__(self, *, node: int, sync_interval_sec: float) -> None:
        if sync_interval_sec <= 0:
            raise ValueError(f"sync_interval_sec must be > 0, got {sync_interval_sec}")
        self._node = int(node)
        self._interval = float(sync_interval_sec)
        self._known: dict[str, VersionLike] = {}
        self._local: set[str] = set()
        self._referenced: set[str] = set()
        self._in_flight: dict[str, FetchJob] = {}
        self._next_try: dict[str, float] = {}
        self.stats = FetchStats()

    # --- inputs ------------------------------------------------------------------------------------------------

    def on_version(self, info: VersionLike) -> bool:
        """Remember ``info`` if it is newer than what is known for its agent, or the same version from a new place
        (new addresses after the learner node rebound its data port). Returns True when it was taken."""
        known = self._known.get(info.agent_id)
        if known is not None:
            if info.version < known.version:
                return False
            if info.version == known.version and (info.node, tuple(info.addresses), info.relay_only) == (
                    known.node, tuple(known.addresses), known.relay_only):
                return False
        self._known[info.agent_id] = info
        return True

    def set_local_agents(self, agents: Iterable[str]) -> None:
        self._local = set(agents)

    def set_referenced(self, agents: Iterable[str]) -> None:
        self._referenced = set(agents)

    def forget(self, agent_id: str) -> None:
        """The agent left the run (``Announce.agents`` status ``removed``)."""
        self._known.pop(agent_id, None)
        self._next_try.pop(agent_id, None)

    # --- decisions ---------------------------------------------------------------------------------------------

    def known(self, agent_id: str) -> VersionLike | None:
        return self._known.get(agent_id)

    def in_flight(self, agent_id: str) -> bool:
        return agent_id in self._in_flight

    def due(self, cached: Mapping[str, int], *, hub_online: bool, now: float) -> list[FetchJob]:
        """Transfers to start now; each returned job is marked in flight (``finished`` / ``failed`` end it)."""
        jobs: list[FetchJob] = []
        for agent_id in sorted(self._known):
            info = self._known[agent_id]
            if agent_id in self._local or info.node == self._node or agent_id in self._in_flight:
                continue
            if now < self._next_try.get(agent_id, -math.inf):
                continue
            have = int(cached.get(agent_id, -1))
            wanted = have < 0 or agent_id in self._referenced
            if not wanted:
                continue
            if hub_online:
                if info.version <= have:
                    continue
                job = FetchJob(agent_id, have, info, "announced")
            else:
                if info.relay_only or not info.addresses:
                    continue
                job = FetchJob(agent_id, have, info, "poll")
            self._in_flight[agent_id] = job
            self._next_try[agent_id] = now + self._interval
            self.stats.jobs += 1
            jobs.append(job)
        return jobs

    def finished(self, agent_id: str, version: int | None, *, cached: int) -> bool:
        """The transfer of ``agent_id`` ended: ``version`` arrived (None: "not modified"); ``cached`` is the cache's
        version before publishing. Returns True when the caller should publish (a newer version arrived)."""
        self._in_flight.pop(agent_id, None)
        if version is None:
            self.stats.not_modified += 1
            return False
        if version <= cached:
            self.stats.redundant += 1
            return False
        self.stats.fetched += 1
        self.stats.history.setdefault(agent_id, []).append(int(version))
        return True

    def failed(self, agent_id: str) -> None:
        """The transfer failed (unreachable, refused, bad digest): retried after one interval."""
        self._in_flight.pop(agent_id, None)
        self.stats.failed += 1
```

Run the two test files: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_counter_book.py tests/unit/test_sp5_fetch_planner.py -q -rw` → 14 passed.

- [ ] **Step 4: Proto additions and `WeightCacheReader.open_latest`**

1. `src/colosseum/net/proto/colosseum_v1.proto` (next free field numbers in each message; keep the T1.2 numbering, append):
```proto
message Heartbeat { ...existing fields...;
  map<string, int64> counters = N;            // node-cumulative counters of this session (CounterBook names, T3.1)
}
message DataAddresses { repeated string data_addresses = 1; }   // the node's data port, announced after binding
message NodeMessage { oneof m { ...existing...; DataAddresses addresses = N; } }
message Welcome { ...existing...; repeated string banned_node_ids = N; }    // data ports refuse these (T6.3 fills)
message Announce { ...existing...; repeated string banned_node_ids = N; }   // the full list when it changes
```
Regenerate with `scripts/gen_proto.sh` (T1.2) and commit the generated files. `PROTOCOL_VERSION` stays `(1, 0)`: the plan's proto is not released before the SP5 merge, the fields are additive. The proto round-trip test of T1.2 (`tests/unit/test_sp5_messages.py` or its name in part 01) gets one assertion per new field (set, serialize, parse, equal).

2. `src/colosseum/node/weights.py` (T1.5), `WeightCacheReader`:
```python
    def open_latest(self, agent_id: str) -> tuple[BinaryIO, dict] | None:
        """The current latest segment of ``agent_id`` opened read-only, with its pointer dict (``segment``,
        ``version``, ``generation``, ``size``, ``digest``, ``teacher_active``); None when the cache holds no latest
        weights of the agent. A pointer whose segment was deleted between the two reads is re-read (bounded
        retries, as ``read_latest``). The caller owns the file object; the open descriptor keeps the bytes readable
        even when the writer deletes the segment afterwards (POSIX). Used by the data port (``GetLatest``) and the
        relay upload (T3.2): they stream the frame without copying it into the Python heap."""
```
implemented on the same pointer/retry helper `read_latest` uses (read the pointer JSON, `open(root / pointer["segment"], "rb")`, on `FileNotFoundError` re-read the pointer, at most the retry bound `read_latest` uses). Add `test_open_latest_returns_the_current_frame_and_survives_a_concurrent_replace` to T1.5's `tests/unit/test_sp5_weight_cache.py` (or its name in part 01): publish v1, `open_latest` → bytes equal the published frame and the pointer's version is 1; publish v2 and v3 (the writer deletes v1) while the v1 file is open → the open file still reads v1's bytes; a fresh `open_latest` returns v3; an unknown agent → None.

- [ ] **Step 5: Write the failing data-port tests**

Create `tests/unit/test_sp5_data_port.py`. It builds a hub identity and two node identities with `net_kit` (T1.3: `hub_identity(tmp_path)`, `node_identity(tmp_path, ca_fp)`, plus `HubIdentity.issue_node_certificate` and `NodeIdentity.save_registration` for a registered node; if `net_kit` already has a `registered_node(tmp_path, hub, name)` helper, use it), a second CA (a second `hub_identity` under another root) for the "foreign certificate" case, and a cache written with `WeightCacheWriter` (frames from `encode_weights` with a 3 MB array so `GetLatest` streams several parts):
```python
"""The node data port (SP5 T3.1, spec 3.1-3.3, 4.5): GetLatest serves the cached frame over mutual TLS, answers
"not modified", refuses foreign, missing and banned certificates, needs the token with --insecure; the port moves
to the next free one; announced addresses; probe and fetch checks."""
from __future__ import annotations

import grpc
import numpy as np
import pytest

from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.grpc_util import channel_options
from colosseum.net.security import (MD_TOKEN, NODE_SERVER_NAME, data_channel_credentials,
                                    data_server_credentials)
from colosseum.node.dataport import (DataServer, FetchError, check_fetched, data_addresses, fetch_latest,
                                     frame_digest, probe_direct)
from colosseum.node.weights import WeightCacheReader, WeightCacheWriter
from net_kit import hub_identity, registered_node      # registered_node: see the note above


def _frame(version, *, teacher_active=False, agent="main"):
    state = {"w": np.full((750_000,), float(version), np.float32)}          # 3 MB: several parts
    return encode_weights(state, WeightsMeta(agent_id=agent, kind="latest", generation=1, policy_version=version,
                                             teacher_active=teacher_active))


def _cache(tmp_path, versions=(3,), teacher_active=False):
    writer = WeightCacheWriter(tmp_path / "shm", "P")
    for v in versions:
        writer.publish_latest("main", _frame(v, teacher_active=teacher_active), version=v, generation=1,
                              teacher_active=teacher_active)
    return WeightCacheReader(tmp_path / "shm", "P")


@pytest.fixture
def pki(tmp_path):
    hub = hub_identity(tmp_path)
    other = hub_identity(tmp_path / "other")                  # another CA
    return {"hub": hub, "other": other, "a": registered_node(tmp_path, hub, "a"),
            "b": registered_node(tmp_path, hub, "b"), "x": registered_node(tmp_path / "other", other, "x")}


def _server(tmp_path, pki, *, banned=(), cache=None):
    srv = DataServer(cache=cache or _cache(tmp_path), own_agents=lambda: {"main"}, max_message_mb=64,
                     insecure=False, token_secret=pki["hub"].token.secret, banned=lambda: set(banned), node=1)
    srv.start(0, credentials=data_server_credentials(pki["a"], pki["hub"].ca_pem), host="127.0.0.1")
    return srv


def _channel(srv, node, ca_pem):
    """A data-port channel with ``node``'s certificate, trusting ``ca_pem`` (the name is always colosseum-node)."""
    creds = data_channel_credentials(node, ca_pem)
    return grpc.secure_channel(f"127.0.0.1:{srv.port}", creds, channel_options(64, server_name=NODE_SERVER_NAME))
```
Tests (each stops its server in `finally`):
- `test_get_latest_streams_the_cached_frame_and_answers_not_modified`: cache v3, v4 with `teacher_active=True`; `fetch_latest(_channel(srv, pki["b"], pki["hub"].ca_pem), "main", -1, max_message_mb=64)` returns `FetchedLatest` with `version == 4`, `generation == 1`, `teacher_active is True`, `frame == _frame(4, teacher_active=True)`, `source == "direct"`; `fetch_latest(..., since_version=4)` returns None; `srv.counters["latest_served"] == 1` and `["latest_not_modified"] == 1`; `fetch_latest(..., "other", -1)` raises `FetchError` whose text contains `NOT_FOUND`.
- `test_peers_without_a_certificate_of_the_ca_or_banned_are_refused`: `_server(..., banned={pki["b"].node_id})` → B's fetch raises `FetchError` matching `PERMISSION_DENIED`; `_channel(srv, pki["x"], pki["other"].ca_pem)` (a certificate of another CA, trusting that CA) and `_channel(srv, pki["x"], pki["hub"].ca_pem)` both raise `FetchError` matching `UNAVAILABLE` (`timeout=3`: the handshake fails); a channel with only `grpc.ssl_channel_credentials(root_certificates=hub.ca_pem)` (no client certificate) raises `FetchError` matching `UNAVAILABLE`; `srv.counters["peer_refused"] >= 1`.
- `test_insecure_data_port_needs_the_token_secret`: `DataServer(..., insecure=True, token_secret="sekret")`, `start(0, credentials=None, host="127.0.0.1")`; a plain channel without metadata → `FetchError` matching `UNAUTHENTICATED`; with `metadata=[(MD_TOKEN, "sekret")]` → `version == 3`.
- `test_a_taken_data_port_moves_to_the_next_free_one`: the first server binds `start(0, ...)` → port P; a second server `start(P, ...)` returns `P + 1` (proves `grpc.so_reuseport` is off: with it on, the second bind of P would succeed); both stop.
- `test_probe_picks_a_live_address_within_the_timeout`: `probe_direct(["127.0.0.1:1", f"127.0.0.1:{srv.port}"], make_channel, timeout=3.0)` returns the live address and an open channel; `probe_direct(["127.0.0.1:1"], ..., timeout=0.5)` and `probe_direct([], ...)` return None.
- `test_data_addresses_start_with_loopback_for_a_loopback_hub`: `data_addresses(47102, "127.0.0.1")[0] == "127.0.0.1:47102"`; no address in `data_addresses(47102, "10.255.255.1")` starts with `127.`; every entry ends with `:47102`.
- `test_a_fetched_frame_is_checked_against_its_metadata`: `check_fetched("main", f, meta, "direct")` with `f = _frame(5)` and `meta = {"version": 5, "generation": 1, "digest": frame_digest(f)}` passes; a flipped byte in the data region → `FetchError` reason `digest`; `meta["version"] = 6` (digest kept) → reason `invalid`; agent `"other"` → `invalid`; `meta` without `generation` → `invalid`; garbage bytes with a matching digest → `invalid` (`peek_meta` fails).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_data_port.py -q` → `ModuleNotFoundError: No module named 'colosseum.node.dataport'`.

- [ ] **Step 6: Write `src/colosseum/node/dataport.py`**

```python
"""The node's data port (SP5 spec 3.1, 4.1, 4.5, 4.7): ``NodeData`` gRPC service, mutual TLS (plain + token with
``--insecure``).

- ``GetLatest(agent, since_version)``: the newest latest weights of an agent whose learner runs on this node,
  straight from the machine's shared-memory cache: one part ``not_modified`` when the cache holds nothing newer
  than ``since_version``, else the weights frame in parts (``grpc_util.send_parts``) whose first part's
  ``meta_json`` is ``{"agent_id", "version", "generation", "size", "digest", "teacher_active"}`` (the cache
  pointer; ``digest`` = blake2b-16 of the frame). Answered without the hub (spec 4.5).
- ``Chunks(stream ChunkFrame)``: frames for this node's learners (T3.2: ``router.LocalDelivery``).

Peers: with TLS the client certificate must be issued by the hub's CA (``require_client_auth``) and its common
name (the node id) must not be banned (``banned()``, from ``Welcome`` / ``Announce``); with ``--insecure`` the
metadata must carry the run token's secret (``x-colosseum-token``). Refusals: ``UNAUTHENTICATED`` /
``PERMISSION_DENIED``; an agent without a learner here: ``NOT_FOUND``.

Binding (spec 3.1): ``distributed.data_port`` on every interface, the next free port when it is taken (up to
``DATA_PORT_TRIES``; ``grpc.so_reuseport`` is off, otherwise a second node on the machine would silently share
the port); port 0 binds an ephemeral port (tests). ``data_addresses`` lists what the node announces.

``probe_direct`` (also used by the chunk router, T3.2) and ``fetch_latest`` are the client side.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import mmap
import socket
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from concurrent import futures
from dataclasses import dataclass
from typing import Any

import grpc

from colosseum.net.frame import FrameError, payload_byte_cap, peek_meta
from colosseum.net.grpc_util import receive_parts, send_parts, server_options
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pbg
from colosseum.net.security import MD_TOKEN

logger = logging.getLogger(__name__)

DATA_PORT_TRIES = 16
DATA_SERVER_THREADS = 64


def frame_digest(frame: bytes | memoryview) -> str:
    """The cache's digest of a frame (``colosseum.node.weights``): blake2b, 16 bytes, hex. (Use T1.5's helper
    instead when ``node.weights`` exports one.)"""
    return hashlib.blake2b(frame, digest_size=16).hexdigest()


def data_addresses(port: int, hub_host: str) -> list[str]:
    """``host:port`` addresses a node announces for its data port: first the local address of the route to the
    hub (a UDP ``connect`` sends nothing), then the machine's other IPv4 addresses; loopback only when the hub
    itself is on loopback (one-machine tests and ``train``'s local node)."""
    hosts: list[str] = []
    loopback_hub = _is_loopback(hub_host)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((hub_host, 9))
            hosts.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            hosts.append(str(info[4][0]))
    except OSError:
        pass
    out: list[str] = []
    for host in hosts:
        if _is_loopback(host) and not loopback_hub:
            continue
        address = f"{host}:{int(port)}"
        if address not in out:
            out.append(address)
    if loopback_hub and f"127.0.0.1:{int(port)}" not in out:
        out.insert(0, f"127.0.0.1:{int(port)}")
    return out


def _is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class DataServer(pbg.NodeDataServicer):
    """The ``NodeData`` servicer and its gRPC server (module docstring). ``cache``: the machine's
    ``WeightCacheReader``; ``own_agents()``: agents whose learner runs on this node (a main-loop snapshot);
    ``delivery``: the T3.2 ``LocalDelivery`` (None: ``Chunks`` is refused with ``UNIMPLEMENTED``)."""

    def __init__(self, *, cache: Any, own_agents: Callable[[], Collection[str]], max_message_mb: int,
                 insecure: bool, token_secret: str, banned: Callable[[], Collection[str]] = frozenset,
                 delivery: Any | None = None, node: int = -1) -> None:
        self._cache = cache
        self._own = own_agents
        self._max_mb = int(max_message_mb)
        self._insecure = bool(insecure)
        self._token_secret = token_secret
        self._banned = banned
        self._delivery = delivery
        self._node = int(node)
        self._server: grpc.Server | None = None
        self.port: int | None = None
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {"latest_served": 0, "latest_not_modified": 0, "peer_refused": 0}

    def set_node(self, node: int) -> None:
        """The node number, known from ``Welcome`` (frames for another node are refused by ``Chunks``)."""
        self._node = int(node)

    # --- lifecycle -----------------------------------------------------------------------------------------------

    def start(self, port: int, *, credentials: grpc.ServerCredentials | None, host: str = "0.0.0.0") -> int:
        """Bind ``port`` (or the next free one, ``DATA_PORT_TRIES`` in all; 0 = ephemeral) and start serving.
        ``credentials`` None means plain (``--insecure``). Raises ``OSError`` when no port could be bound."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=DATA_SERVER_THREADS, thread_name_prefix="data"),
                             options=server_options(self._max_mb))
        pbg.add_NodeDataServicer_to_server(self, server)
        candidates = [0] if port == 0 else range(int(port), int(port) + DATA_PORT_TRIES)
        bound = 0
        for candidate in candidates:
            address = f"{host}:{candidate}"
            try:
                bound = (server.add_insecure_port(address) if credentials is None
                         else server.add_secure_port(address, credentials))
            except RuntimeError:            # grpcio raises on a failed bind ("Failed to bind to address ...")
                continue
            if bound:
                break
        if not bound:
            raise OSError(f"no free data port in {port}..{int(port) + DATA_PORT_TRIES - 1} on {host}")
        if port and bound != port:
            logger.info(f"Data port {port} is taken; listening on {bound}")
        server.start()
        self._server, self.port = server, bound
        return bound

    def stop(self, grace: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace).wait(grace + 1.0)
            self._server = None

    # --- NodeData ------------------------------------------------------------------------------------------------

    def GetLatest(self, request: pb.LatestRequest, context: grpc.ServicerContext) -> Iterator[pb.BlobPart]:  # noqa: N802
        self._check_peer(context)
        agent_id = request.agent_id
        if agent_id not in set(self._own()):
            context.abort(grpc.StatusCode.NOT_FOUND, f"node {self._node} runs no learner of agent {agent_id!r}")
        opened = self._cache.open_latest(agent_id)
        if opened is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"no weights of agent {agent_id!r} in the cache yet")
        f, pointer = opened
        with f:
            meta = {"agent_id": agent_id, **{k: pointer[k] for k in ("version", "generation", "size", "digest",
                                                                     "teacher_active")}}
            if int(pointer["version"]) <= int(request.since_version):
                self._count("latest_not_modified")
                yield pb.BlobPart(not_modified=True, last=True, meta_json=json.dumps(meta))
                return
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                yield from send_parts(memoryview(mm), meta_json=json.dumps(meta))
        self._count("latest_served")

    def Chunks(self, request_iterator: Iterable[pb.ChunkFrame], context: grpc.ServicerContext) -> pb.ChunkSummary:  # noqa: N802
        self._check_peer(context)
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "chunks over the data port arrive with T3.2")

    # --- internals -----------------------------------------------------------------------------------------------

    def _check_peer(self, context: grpc.ServicerContext) -> None:
        if self._insecure:
            md = dict(context.invocation_metadata())
            if md.get(MD_TOKEN) != self._token_secret:
                self._count("peer_refused")
                context.abort(grpc.StatusCode.UNAUTHENTICATED,
                              "data port: missing or wrong token secret (x-colosseum-token)")
            return
        names = context.auth_context().get("x509_common_name") or []
        node_id = names[0].decode() if names else ""
        if not node_id:
            self._count("peer_refused")
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "data port: no node certificate")
        if node_id in set(self._banned()):
            self._count("peer_refused")
            context.abort(grpc.StatusCode.PERMISSION_DENIED, f"data port: node {node_id} is banned")

    def _count(self, name: str) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + 1


def probe_direct(addresses: Sequence[str], make_channel: Callable[[str], grpc.Channel],
                 timeout: float) -> tuple[str, grpc.Channel] | None:
    """The first of ``addresses`` whose channel becomes READY within ``timeout`` seconds (all probed at once;
    READY includes the TLS handshake); the other channels are closed. None when none answers (spec 4.8: then the
    relay). ``grpc.channel_ready_future`` returns a ``grpc.Future``, not a ``concurrent.futures`` one: completion
    is signalled through ``add_done_callback``."""
    if not addresses:
        return None
    channels = {a: make_channel(a) for a in addresses}
    changed = threading.Event()
    ready = {a: grpc.channel_ready_future(ch) for a, ch in channels.items()}
    for fut in ready.values():
        fut.add_done_callback(lambda _f: changed.set())
    winner = None
    deadline = time.monotonic() + timeout
    while winner is None:
        for address, fut in ready.items():
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                winner = address
                break
        remaining = deadline - time.monotonic()
        if winner is not None or remaining <= 0 or all(f.done() for f in ready.values()):
            break
        changed.wait(remaining)
        changed.clear()
    for address, fut in ready.items():
        if address != winner:
            fut.cancel()
            channels[address].close()
    return None if winner is None else (winner, channels[winner])


class FetchError(Exception):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class FetchedLatest:
    agent_id: str
    version: int
    generation: int
    teacher_active: bool
    frame: bytes
    source: str            # "direct" | "relay"


def check_fetched(agent_id: str, frame: bytes, meta: dict, source: str) -> FetchedLatest:
    """Hostile-input checks of a fetched latest frame (spec 3.4): the digest of the bytes, the frame header's agent
    and version against the reply's metadata."""
    try:
        version, generation = int(meta["version"]), int(meta["generation"])
        digest, teacher_active = str(meta["digest"]), bool(meta.get("teacher_active", False))
    except (KeyError, TypeError, ValueError) as e:
        raise FetchError("invalid", f"metadata {meta!r}") from e
    if frame_digest(frame) != digest:
        raise FetchError("digest", f"agent {agent_id} v{version}")
    try:
        kind, header = peek_meta(frame)
    except FrameError as e:
        raise FetchError("invalid", f"frame header: {e.reason}") from e
    if kind != "weights" or header.get("agent_id") != agent_id or int(header.get("policy_version", -1)) != version:
        raise FetchError("invalid", f"frame of {header.get('agent_id')!r} v{header.get('policy_version')} for "
                                    f"{agent_id!r} v{version}")
    return FetchedLatest(agent_id, version, generation, teacher_active, frame, source)


def read_latest_reply(parts: Iterator[pb.BlobPart], agent_id: str, *, max_message_mb: int,
                      source: str) -> FetchedLatest | None:
    """A ``GetLatest`` / ``Download(kind="latest")`` reply: None for "not modified", else the checked frame.
    Raises ``FetchError`` (``grpc.RpcError`` and malformed parts included)."""
    try:
        first = next(parts)
        if first.not_modified:
            return None
        frame, meta_json = receive_parts(_chain(first, parts), cap=payload_byte_cap(max_message_mb))
    except grpc.RpcError as e:
        raise FetchError("rpc", f"{e.code()}: {e.details()}") from e
    except StopIteration as e:
        raise FetchError("truncated") from e
    except ValueError as e:            # receive_parts: size cap, offsets, digest (T2.3 raises a ValueError kind)
        raise FetchError("blob", str(e)) from e
    try:
        meta = json.loads(meta_json)
    except ValueError as e:
        raise FetchError("invalid", "metadata is not JSON") from e
    return check_fetched(agent_id, bytes(frame), meta, source)


def fetch_latest(channel: grpc.Channel, agent_id: str, since_version: int, *, max_message_mb: int,
                 metadata: Sequence[tuple[str, str]] = (), timeout: float = 60.0) -> FetchedLatest | None:
    """``GetLatest`` on a node's data port. None: not modified. Raises ``FetchError`` (unreachable, refused,
    truncated, wrong digest, wrong frame)."""
    stub = pbg.NodeDataStub(channel)
    try:
        parts = stub.GetLatest(pb.LatestRequest(agent_id=agent_id, since_version=int(since_version)),
                               metadata=tuple(metadata), timeout=timeout)
    except grpc.RpcError as e:
        raise FetchError("rpc", f"{e.code()}: {e.details()}") from e
    return read_latest_reply(parts, agent_id, max_message_mb=max_message_mb, source="direct")


def _chain(first: pb.BlobPart, rest: Iterator[pb.BlobPart]) -> Iterator[pb.BlobPart]:
    yield first
    yield from rest
```
Notes for the implementer: `send_parts` must accept any bytes-like object (here a `memoryview` of an `mmap`; slicing it copies one `PART_BYTES` part at a time) — if T2.3's `send_parts` only takes `bytes | Path`, widen its type to `bytes | memoryview | Path` (its body already slices). `server_options` (T1.3) must contain `("grpc.so_reuseport", 0)`; if it does not, add it there (the hub port needs it as much: two hubs on one machine, contract note 15).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_data_port.py -q -rw` → 7 passed.

- [ ] **Step 7: Node: data port, fetch section, counters (`src/colosseum/node/agent.py`)**

Edit points in `NodeAgent` (T2.4 names; adapt to the code as it is):

1. `NodeOptions` gains `data_addresses: tuple[str, ...] | None = None` (announce these instead of the probed ones: a port forward or a public address; tests put a `TcpFaultProxy` there in T3.2). `train`'s local node keeps `data_port=None` unless `train --listen` (overview R12), so it has no data port.
2. After the first `Welcome` (the node then knows its number, `config.distributed` and whether the session is TLS): if `options.data_port is not None` and the data port is not running, build `DataServer(cache=self._cache_reader, own_agents=lambda: self._local_learner_agents, max_message_mb=cfg.distributed.max_message_mb, insecure=options.insecure, token_secret=options.token.secret, banned=lambda: self._banned, node=number)` and `start(port, credentials=None if options.insecure else data_server_credentials(self._identity, ca_pem))` where `port = cfg.distributed.data_port` when `options.data_port == -1` (`join`'s default, contract note 9: the port comes from the hub's config) and `options.data_port` otherwise (0 = ephemeral: tests). A bind failure (`OSError`) is one WARNING ("no data port: other machines reach this node's learners only through the hub") and the node runs without one. Then `addresses = list(options.data_addresses or data_addresses(bound_port, hub_host))` and send `NodeMessage(addresses=DataAddresses(data_addresses=addresses))`; every later `Hello` carries them in `Hello.data_addresses`. The data port lives as long as the node process (across sessions and runs of this hub); `NodeAgent` stops it in its `finally`.
3. `self._fetch = FetchPlanner(node=number, sync_interval_sec=cfg.rollout.weight_sync_interval_sec)` on the first `Welcome` (a later `Welcome` of another run makes a new one). `_on_welcome` and `_on_announce` call `self._fetch.on_version(version_from_proto(v))` for every `versions` entry, `forget` for agents whose `AgentState.status` is `removed`, and store `banned_node_ids` into `self._banned` (a frozenset; `Welcome` replaces it, `Announce` replaces it when the field is non-empty or `banned_node_ids` presence is signalled — use "replace when the message carries the field": an `Announce` without bans leaves the set; T6.3 sends the full list on every change).
4. `self._fetch.set_local_agents(...)` whenever this node's learners change (start/stop commands).
5. A new periodic step in the main loop, `_pump_fetches(now)`, at most every 0.1 s: `cached = {agent: v for agent, (v, _gen) in ((a, self._cache_reader.latest_version(a)) for a in known) if v is not None}`; `referenced` = every agent named by a `CacheKey("latest", agent)` of a live cell's current or pending lineup (the R11 keys T2.4 already computes per grant); `self._fetch.set_referenced(referenced)`; `jobs = self._fetch.due(cached, hub_online=self._session.connected, now=now)`; each job goes to the transfer pool as `_fetch_job(job)`.
6. `_fetch_job(job)` (transfer thread, no main-loop state): pick a channel to the learner node — `self._data_channels.get(job.source)`: a small cache `node number -> (address, channel)` guarded by a lock; on a miss `probe_direct(job.source.addresses, make_channel, cfg.distributed.connect_timeout_sec)` with `make_channel = lambda a: grpc.secure_channel(a, data_channel_credentials(identity, ca_pem), channel_options(max_mb, server_name=NODE_SERVER_NAME))` (insecure: `grpc.insecure_channel(a, channel_options(max_mb))` and `metadata=[(MD_TOKEN, token.secret)]`); no address answers → post `LatestFetchFailed(job.agent_id, "unreachable")` (T3.2 adds the relay before this). Then `fetch_latest(channel, job.agent_id, job.since_version, max_message_mb=..., metadata=...)` → post `LatestFetched(fetched)` (None → `LatestFetched(agent_id, None)`); `FetchError` → drop the cached channel, post `LatestFetchFailed(agent_id, e.reason)`.
7. `_on_latest_fetched(ev)` (main loop): `cached = self._cache_reader.latest_version(agent)`; `if self._fetch.finished(agent, ev.version, cached=cached_v)`: `self._cache_writer.publish_latest(agent, ev.frame, version=ev.version, generation=ev.generation, teacher_active=ev.teacher_active)` (the frame as received: byte-for-byte the learner's segment) and `self._counters["weight_fetch_bytes"] += len(ev.frame)`; `CacheFullError` → the ERROR of T1.5 with its hint (`docker --shm-size`, `emptyDir: Memory`), `self._fetch.failed(agent)` and the T2.4 cache-full handling (the node takes no new work). The workers see the new pointer at their next `weight_sync_interval_sec` poll (`WorkerIO.poll_weights` reads it with `teacher_active`).
8. `_on_latest_fetch_failed(ev)`: `self._fetch.failed(agent)`; one WARNING per (agent, reason) per session ("weights of agent X from node N: <reason>; retried every <interval> s"); every pending grant waiting for `CacheKey("latest", agent)` takes T2.4's failed-fetch path (grant dropped, old lineup kept, `lineup_fetch_failed += 1`, re-requested at the next pass).
9. `Heartbeat.counters` = the node's cumulative counters for this session: `self._fetch.stats.as_counters()` + `weight_fetch_bytes` + `lineup_fetch_failed` (T2.4's counter; if T2.4 keeps it elsewhere, read it from there) + the data port's `latest_served` / `latest_not_modified` / `peer_refused` prefixed `data_port.`. A new session (a new node process) starts from zero; the planner's stats belong to the process, so a reconnect within the same process keeps counting (session ids change only per process, overview T2.3 `Hello.session_id`).
10. `NodeAgent.counters() -> dict[str, int]` (a copy of the values of item 9, published by the main loop each pass into an attribute and returned as is: thread-safe for tests) and `NodeAgent.fetch_history() -> dict[str, list[int]]` (a copy of `self._fetch.stats.history`, published the same way).

- [ ] **Step 8: Hub: version fan-out, data addresses, the counter book (`src/colosseum/hub/state.py`, `hub/hub.py`, `hub/server.py`)**

1. `src/colosseum/hub/state.py`, after `VersionRegistry`:
```python
def version_source(info: VersionInfo, entry: NodeEntry) -> VersionInfo:
    """``info`` as announced to the other nodes (spec 4.5 "где взять"): the announcing node's number and data
    addresses, whatever the node wrote itself; ``relay_only`` when it has no data address (``train``'s local node
    without ``--listen``, a node whose data port could not be bound)."""
    return dataclasses.replace(info, node=int(entry.number), addresses=tuple(entry.data_addresses),
                               relay_only=not entry.data_addresses)
```
2. `hub/server.py`: the session servicer forwards a `NodeMessage` with `addresses` like every other oneof (`NodeMessageIn`); nothing else changes there in this step.
3. `hub/hub.py`:
   - `_on_version(...)`: after `VersionRegistry.announce(...)` returned `"accepted"`, replace the stored latest entry by `version_source(info, node_entry)` (so `Welcome` and announcements carry where to fetch) and put it into `self._pending_versions[agent_id]` (a dict: the newest per agent wins).
   - `_flush_announcements()` at the end of every main-loop pass: when `self._pending_versions` is not empty, send one `Announce(versions=[version_to_proto(v) for v in pending.values()])` to every online session (T2.5's `Announce` sender; merge with its evictions / env-step fields if it batches them) and clear the dict.
   - `_on_data_addresses(node_id, session_id, msg)`: `NodeEntry.data_addresses = tuple(msg.addresses.data_addresses)` (bounded: at most 16 entries of at most 255 characters each, else dropped with one WARNING per node, counted `invalid` in the book as `node_messages.invalid`); then for every agent whose latest entry has `node == entry.number`, re-store `version_source(latest, entry)` and queue it in `_pending_versions` (nodes learn the new addresses at once).
   - `Welcome.versions` = `[version_to_proto(v) for v in registry latest entries]` (already filled by `version_source`); `Welcome.banned_node_ids` = `[]` until T6.3.
   - `self._book = CounterBook()` in `Hub.__init__` (T4.3 restores it from `hub_state.json` on `--resume`: `CounterBook.from_state(counters.get("book"))`); `_on_heartbeat(...)`: `self._book.report(f"{node_id}/{session_id}", dict(msg.heartbeat.counters))`.
   - `hub_state.json` (T2.5 writes it on every metrics tick): `counters` gains `"book": self._book.state()` and `lineup_fetch_failed` = `self._book.get("lineup_fetch_failed")` (the contract key already listed in `HubStateData.counters`; T3.2 fills `relay_bytes` and `dropped_chunks` the same way). `metrics.jsonl` keys stay T7.1's.

Create `tests/unit/test_sp5_version_source.py`:
- `test_announced_versions_carry_the_announcing_nodes_addresses`: a `VersionInfo(agent_id="main", generation=1, version=4, size=9, digest="d", node=99, addresses=("6.6.6.6:1",))` and a `NodeEntry` with `number=2`, `data_addresses=("10.0.0.2:47102",)` → `version_source` gives `node == 2`, `addresses == ("10.0.0.2:47102",)`, `relay_only is False`, every other field equal.
- `test_a_node_without_a_data_address_is_relay_only`: `data_addresses=()` → `relay_only is True`, `addresses == ()`.
(Build `NodeEntry` with the fields of the contract; add a `make_node_entry(**kw)` helper to `hub_kit` if T2.3 has none.)

- [ ] **Step 9: `tests/fault_proxy.py`, `tests/cluster_kit.py` and the fetch integration tests**

Create `tests/fault_proxy.py` (add `"fault_proxy"` to `known-first-party`; the overview assigns it to T3.2, T3.1 needs it first, contract note 21):
```python
"""TCP fault injection between two local endpoints (SP5 T3.1, used by T3.1 and T3.2; test support, imported by bare name).

``TcpFaultProxy(("127.0.0.1", port))`` listens on ``127.0.0.1:<ephemeral>`` (``proxy.address``) and forwards
every accepted connection to the target, both directions, through its own threads. The target may be set later
(``TcpFaultProxy()`` then ``set_target``: a node can announce the proxy's address before its own data port is
bound); until then connections are closed at once. Faults:
- ``delay(sec)``: every forwarded read waits ``sec`` first (both directions);
- ``cut(sec)``: every open connection is closed now and new ones are accepted and closed at once for ``sec``
  seconds (``time.monotonic``), then forwarding resumes by itself;
- ``drop()``: like ``cut`` without an end (until ``restore()``);
- ``stall()``: connections are accepted but nothing is forwarded (a TLS handshake never completes: the
  "unreachable within connect_timeout_sec" case of spec 4.8) until ``restore()``;
- ``restore()``: normal forwarding (also ends a delay).
``connections_seen`` counts accepted connections, ``bytes_forwarded`` the bytes moved. ``close()`` stops
everything; use it as a context manager. No wall clock, no sleeps longer than 50 ms per loop.
"""

from __future__ import annotations

import socket
import threading
import time

_POLL_SEC = 0.05
_BUF = 1 << 16


class TcpFaultProxy:
    def __init__(self, target: tuple[str, int] | None = None, *, host: str = "127.0.0.1") -> None:
        self._target = target
        self._lock = threading.Lock()
        self._mode = "normal"            # normal | cut | stall
        self._cut_until = 0.0
        self._delay = 0.0
        self._conns: list[socket.socket] = []
        self._closed = threading.Event()
        self.connections_seen = 0
        self.bytes_forwarded = 0
        self._server = socket.create_server((host, 0))
        self._server.settimeout(_POLL_SEC)
        self.port = self._server.getsockname()[1]
        self.address = f"{host}:{self.port}"
        self._acceptor = threading.Thread(target=self._accept_loop, name="fault-proxy", daemon=True)
        self._acceptor.start()

    def set_target(self, target: tuple[str, int]) -> None:
        with self._lock:
            self._target = target

    # --- faults ------------------------------------------------------------------------------------------------

    def delay(self, sec: float) -> None:
        with self._lock:
            self._delay = float(sec)

    def cut(self, sec: float) -> None:
        with self._lock:
            self._mode, self._cut_until = "cut", time.monotonic() + float(sec)
        self._close_all()

    def drop(self) -> None:
        with self._lock:
            self._mode, self._cut_until = "cut", float("inf")
        self._close_all()

    def stall(self) -> None:
        with self._lock:
            self._mode = "stall"
        self._close_all()

    def restore(self) -> None:
        with self._lock:
            self._mode, self._cut_until, self._delay = "normal", 0.0, 0.0

    def close(self) -> None:
        self._closed.set()
        self._close_all()
        self._server.close()
        self._acceptor.join(timeout=2)

    def __enter__(self) -> TcpFaultProxy:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- internals -----------------------------------------------------------------------------------------------

    def _state(self) -> tuple[str, float]:
        with self._lock:
            if self._mode == "cut" and time.monotonic() >= self._cut_until:
                self._mode = "normal"
            return self._mode, self._delay

    def _close_all(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        for s in conns:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            s.close()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections_seen += 1
            mode, _ = self._state()
            with self._lock:
                target = self._target
            if mode == "cut" or target is None:
                client.close()
                continue
            if mode == "stall":
                with self._lock:
                    self._conns.append(client)          # held open, never forwarded
                continue
            try:
                upstream = socket.create_connection(target, timeout=2.0)
            except OSError:
                client.close()
                continue
            with self._lock:
                self._conns += [client, upstream]
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        src.settimeout(_POLL_SEC)
        try:
            while not self._closed.is_set():
                try:
                    data = src.recv(_BUF)
                except TimeoutError:
                    continue
                if not data:
                    break
                mode, delay = self._state()
                if mode != "normal":
                    break
                if delay > 0:
                    time.sleep(delay)
                dst.sendall(data)
                self.bytes_forwarded += len(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass
```

Create `tests/unit/test_sp5_fault_proxy.py`:
```python
"""TcpFaultProxy (SP5 T3.1 test support): forwards, delays, cuts for a while, drops, stalls and restores."""
from __future__ import annotations

import socket
import threading
import time

import pytest

from fault_proxy import TcpFaultProxy


@pytest.fixture
def echo():
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.05)
    stop = threading.Event()

    def serve(conn):
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                conn.sendall(data)

    def loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:                                          # closed at teardown
                return
            threading.Thread(target=serve, args=(conn,), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    yield server.getsockname()[1]
    stop.set()
    server.close()


def _roundtrip(port, payload=b"ping", timeout=1.0):
    """The echoed bytes; b"" when the proxy closed the connection; None on a refusal, a reset or a timeout."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
            s.sendall(payload)
            return s.recv(4096)
    except OSError:
        return None


def test_forwards_and_counts(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        assert _roundtrip(proxy.port) == b"ping"
        assert proxy.connections_seen == 1 and proxy.bytes_forwarded >= 8


def test_cut_closes_connections_and_ends_by_itself(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        s = socket.create_connection(("127.0.0.1", proxy.port), timeout=1.0)
        s.sendall(b"a")
        assert s.recv(10) == b"a"
        proxy.cut(0.5)
        assert s.recv(10) == b""                                     # the open connection was closed
        s.close()
        assert not _roundtrip(proxy.port)                            # closed at once while the cut lasts
        time.sleep(0.6)
        assert _roundtrip(proxy.port) == b"ping"                     # and restored by itself


def test_drop_and_stall_last_until_restore(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        proxy.drop()
        assert not _roundtrip(proxy.port)
        proxy.stall()
        assert _roundtrip(proxy.port, timeout=0.3) is None            # accepted, never answered: timeout
        proxy.restore()
        assert _roundtrip(proxy.port) == b"ping"


def test_delay_slows_every_read(echo):
    with TcpFaultProxy(("127.0.0.1", echo)) as proxy:
        proxy.delay(0.2)
        t0 = time.monotonic()
        assert _roundtrip(proxy.port) == b"ping"
        assert time.monotonic() - t0 >= 0.35                         # two forwarded reads, 0.2 s each


def test_a_target_set_later_is_used_from_then_on(echo):
    with TcpFaultProxy() as proxy:
        assert not _roundtrip(proxy.port)                            # no target yet: closed at once
        proxy.set_target(("127.0.0.1", echo))
        assert _roundtrip(proxy.port) == b"ping"
```
Run it: 5 passed.

Create `tests/cluster_kit.py` (add `"cluster_kit"` to `known-first-party` in `pyproject.toml`):
```python
"""Localhost clusters for SP5 tests (T3.1; later tasks extend): an in-process hub (``Hub.launch`` in a thread of
the test process: hub logic is inspectable, exit codes are returned) and ``NodeAgent``s in threads, each with its
own identity root, log dir and data port (port 0); their children (workers, learners) are real processes, so a
cluster test counts them against the "2 loaded CPU processes" rule. Waits are on counters and files, never on
wall-clock values. Every helper stops what it started in ``finally`` (``cluster``).

The hub listens on 127.0.0.1 (``HubOptions(mode="hub", listen="127.0.0.1", port=0)``); nodes find it through
``runs/<name>/hub.json``; with ``insecure=False`` (default) the hub port is TLS and data ports are mutual TLS, as
in production. ``cluster_config`` is the tic-tac-toe example with the TINY sizes of ``cli_runner`` and fast
distributed timings.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli_runner import TINY, TTT_CONFIG, wait_for

from colosseum.core.config import ColosseumConfig, apply_overrides, load_config
from colosseum.core.run_dir import RunDir
from colosseum.hub.hub import Hub, HubOptions
from colosseum.hub.registry import read_hub_json
from colosseum.net.security import HubIdentity, Token
from colosseum.node.agent import NodeAgent, NodeOptions

CLUSTER_TINY: dict[str, str] = {
    **TINY,
    "distributed.heartbeat_sec": "0.25",
    "distributed.node_lost_after_sec": "5.0",
    "distributed.connect_timeout_sec": "1.0",
    "distributed.results_flush_sec": "0.1",
    "distributed.shutdown_grace_sec": "5.0",
}


def wait_until(predicate: Callable[[], bool], timeout: float, interval: float = 0.1) -> bool:
    return wait_for(predicate, timeout, interval)


def cluster_config(base: Path = TTT_CONFIG, overrides: Mapping[str, Any] | None = None) -> ColosseumConfig:
    """``base`` with ``CLUSTER_TINY`` and ``overrides`` (``--set`` style keys and YAML-literal values)."""
    sets = {**CLUSTER_TINY, **{k: str(v) for k, v in (overrides or {}).items()}}
    return apply_overrides(load_config(str(base)), [f"{k}={v}" for k, v in sets.items()])


@dataclass
class HubThread:
    hub: Hub
    thread: threading.Thread
    root: Path                       # runs/<name>
    address: str                     # host:port of the hub port
    token: Token
    insecure: bool
    exit_codes: list[int] = field(default_factory=list)

    def stop(self, timeout: float = 30.0) -> int | None:
        self.hub.request_stop()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the hub did not stop"
        return self.exit_codes[0] if self.exit_codes else None


def start_hub(tmp_path: Path, config: ColosseumConfig, *, insecure: bool = False, name: str = "cluster") -> HubThread:
    run_dir = RunDir.create(config, tmp_path / "runs" / name)           # T2.5 creates run dirs this way
    hub = Hub(config, run_dir, options=HubOptions(mode="hub", listen="127.0.0.1", port=0, insecure=insecure))
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(hub.launch()), name="hub", daemon=True)
    thread.start()
    assert wait_until(lambda: read_hub_json(run_dir.root) is not None or not thread.is_alive(), 30)
    info = read_hub_json(run_dir.root)
    assert info is not None, f"the hub exited with {codes}"
    token = HubIdentity.load_or_create(Path(info.identity_dir)).token
    return HubThread(hub, thread, run_dir.root, info.addresses[0], token, insecure, codes)


@dataclass
class NodeThread:
    agent: NodeAgent
    thread: threading.Thread
    name: str
    exit_codes: list[int] = field(default_factory=list)

    def stop(self, timeout: float = 30.0) -> int | None:
        self.agent.request_stop()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), f"node {self.name} did not stop"
        return self.exit_codes[0] if self.exit_codes else None


def start_node(tmp_path: Path, hub: HubThread, name: str, *, workers: int = 0, cpu_learners: int = 0,
               evaluators: int = 0, no_learners: bool = False, data_port: int | None = 0,
               data_addresses: Sequence[str] | None = None, once: bool = True) -> NodeThread:
    options = NodeOptions(hub_address=hub.address, token=hub.token, insecure=hub.insecure, workers=workers,
                          evaluators=evaluators, cpu_learners=cpu_learners, no_learners=no_learners, name=name,
                          once=once, identity_root=tmp_path / "nodes" / name, log_dir=tmp_path / "logs" / name,
                          data_port=data_port,
                          data_addresses=None if data_addresses is None else tuple(data_addresses))
    agent = NodeAgent(options)
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(agent.run()), name=f"node-{name}", daemon=True)
    thread.start()
    return NodeThread(agent, thread, name, codes)


@contextmanager
def cluster(tmp_path: Path, config: ColosseumConfig, nodes: Sequence[Mapping[str, Any]], *,
            insecure: bool = False, learner_first: str | None = None,
            timeout: float = 60.0) -> Iterator[tuple[HubThread, list[NodeThread]]]:
    """A hub and ``nodes`` (``start_node`` keywords + ``name``). With ``learner_first=<agent>``, every node after
    the first starts only once the hub registry has a version of that agent (its learner runs on the first node:
    the hub of T2.5 places learners on the first node with a learner slot)."""
    hub = start_hub(tmp_path, config, insecure=insecure)
    started: list[NodeThread] = []
    try:
        for i, spec in enumerate(nodes):
            if i == 1 and learner_first is not None:
                assert wait_until(lambda: hub_version(hub, learner_first) is not None, timeout), \
                    f"no version of {learner_first} announced"
            started.append(start_node(tmp_path, hub, **dict(spec)))
        yield hub, started
    finally:
        for node in started:
            node.agent.request_stop()
        try:
            hub.stop()
        finally:
            for node in started:
                node.stop()


def read_hub_state(hub: HubThread) -> dict:
    path = hub.root / "hub_state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def hub_version(hub: HubThread, agent_id: str) -> int | None:
    """The newest version of ``agent_id`` the hub accepted, from ``hub_state.json`` (written every metrics tick)."""
    latest = (read_hub_state(hub).get("versions", {}).get(agent_id) or {}).get("latest")
    return None if not latest else int(latest["version"])


def journal_results_by_node(hub: HubThread) -> dict[int, int]:
    """Training results in ``matches.db`` per node number, from the uid segment ``n<node>/`` (spec 4.3)."""
    db = hub.root / "matches.db"
    if not db.exists():
        return {}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        uids = [u for (u,) in con.execute("SELECT uid FROM matches WHERE source = 'train'")]
    out: dict[int, int] = {}
    for uid in uids:
        part = next((p for p in uid.split("/") if p.startswith("n") and p[1:].isdigit()), None)
        if part is not None:
            out[int(part[1:])] = out.get(int(part[1:]), 0) + 1
    return out
```
Adapt the imports to T2.5's names (`RunDir.create` signature, `HubOptions` fields, how `Hub.launch` reports its exit code); the docstring's rules are binding. If `Hub.launch` or `NodeAgent.run` installs signal handlers unconditionally, guard them with `threading.current_thread() is threading.main_thread()` (an edit of T2.5's `Hub.launch` / T2.4's `NodeAgent.run`; both keep their behaviour in the main thread) and add `test_hub_launch_off_the_main_thread_installs_no_signal_handlers` to `tests/unit/test_sp5_cluster_kit.py` (the hub thread runs, `signal.getsignal(SIGINT)` is unchanged, `request_stop` returns exit code 0).

Create `tests/integration/test_sp5_node_fetch.py` — node logic with `node_kit.FakeHub` (T2.4: an in-process hub servicer with a scripted `Welcome`, grants, announcements, a results recorder; adapt the calls to its API). Common setup: a tic-tac-toe `cluster_config()`; a "learner node" emulated by the test: a `WeightCacheWriter` on a second shm prefix plus a real `DataServer(insecure=True, token_secret=fake_hub.token.secret, own_agents={"main"}, node=7)` started on port 0 (the frames: `encode_weights(model.state_dict() as numpy, WeightsMeta(kind="latest", agent_id="main", generation=1, policy_version=v, teacher_active=...))` for the tic-tac-toe model built from the config, so the node's workers can load them); the FakeHub's `Welcome` lists agent `main` with `VersionInfo(node=7, addresses=(f"127.0.0.1:{port}",), version=v0)` and grants every cell the lineup `main@latest` (collect False: no learner on this node) vs the scripted anchor `random` of the config. The node under test runs `NodeAgent` in a thread with `insecure=True` and the given `workers`.
- `test_one_download_per_version_whatever_the_number_of_workers` (node with **2 workers**: the only loaded processes): publish v1, `FakeHub.announce(version=1)`; wait until the FakeHub recorded results whose `main` seats have `version_end == 1`; then for v in 2..5: publish, announce, and wait until a result with `version_end == v` arrived (each wait ≤ 3 × `weight_sync_interval_sec` + 10 s). Assert: `node.agent.fetch_history()["main"]` is strictly increasing, ends with 5 and has no repeated version; `data_server.counters["latest_served"] == len(history)` (every download was one transfer, whatever the 2 workers did); the last `Heartbeat.counters` the FakeHub recorded has `weight_fetches == len(history)` and `weight_fetch_redundant == 0`; both workers (results from `w0_` and `w1_` match ids) played `version_end == 5`.
- `test_without_the_hub_versions_come_from_get_latest` (1 worker): after v1 arrived (as above), `FakeHub.close_sessions()` and refuse reconnects (or stop the FakeHub server); publish v2 without any announcement; assert within 3 × `weight_sync_interval_sec` + 5 s the node's cache pointer (`WeightCacheReader(node shm root, node prefix).latest_version("main")`) is `(2, 1)`; `data_server.counters["latest_not_modified"] >= 1` (polls before v2 existed); `node.agent.counters()["weight_fetches"] == 2`.
- `test_teacher_active_off_reaches_the_remote_cache` (1 worker): publish v1 with `teacher_active=True`, announce, wait for it in the node's cache, then v2 with `teacher_active=False`; assert the node cache's `read_latest("main").meta.teacher_active is False` and the pointer's `teacher_active` is False (the node publishes the flag it received; workers read it through `WorkerIO.poll_weights`, T2.2).
- `test_a_failed_first_fetch_holds_the_lineup_and_counts_lineup_fetch_failed` (1 worker): the `Welcome` names the learner node's address behind a `fault_proxy.TcpFaultProxy` in `drop()` (Step 9 creates `tests/fault_proxy.py`); assert the FakeHub records a `Heartbeat.counters` with `lineup_fetch_failed >= 1` and no result arrives while the drop lasts; `proxy.restore()` → results with `version_end == 1` arrive and `fetch_history() == {"main": [1]}`.
- `test_a_full_cache_is_an_error_with_the_hint_and_counted` (no worker needed: `workers=0` with a FakeHub that grants nothing): set `COLOSSEUM_SHM_DIR` to a directory whose free space the test controls by monkeypatching T1.5's free-space probe (the function `WeightCacheWriter` calls before a write) to report 1 MiB; announce v1; assert one ERROR in the node log containing `--shm-size` and `emptyDir`, `counters()["weight_fetch_failed"] >= 1`, nothing published.

Create `tests/integration/test_sp5_weights_cluster.py` — real hub and two real nodes (`cluster_kit.cluster`, TLS on: hub port TLS, data ports mutual TLS):
- `test_a_second_node_plays_with_the_learner_nodes_weights_and_the_frozen_blob`: config = `cluster_config()` with a frozen agent `frozen` (the tic-tac-toe model's state dict written to `tmp_path / "frozen.pt"` with the config's `networks`; `agents: {main: {}, frozen: {kind: frozen, path: ..., networks: ...}}`, `matchmaking.opponents: {latest: 0, snapshots: 0, rivals: 0, anchors: 1}`, `matchmaking.anchors: [frozen]`) and `training.total_timesteps` large (the test stops the hub). Nodes: `a` (`cpu_learners=1`, `workers=0`: its learner is the only loaded process there) then `b` (`workers=1`, `no_learners=True`), `learner_first="main"`. Wait until `journal_results_by_node(hub)` has ≥ 20 results of node b's number. Assert: `b.agent.fetch_history() == {"main": [0]}` (the learner's initial weights; no collection until T3.2, so the version stays 0); B's cache holds `CacheKey("blob", "frozen", <blob id from Welcome.fixed>)` and its `meta.architecture` equals the frozen entry's `networks` (`WeightCacheReader.get(...)`); `read_hub_state(hub)["counters"]["book"]["totals"]["weight_fetches"] == 1` after the next metrics tick (wait for it); node a's data port `counters["latest_served"] == 1`; every journaled result of node b has a `frozen` seat with network `fixed`.
- `test_teacher_and_init_blobs_reach_a_second_node`: config with `agents.main.kickstart.teacher` = a neural teacher `.pt` (the frozen file above, kind `teacher` blob) and `agents.main.init.from` = the same `.pt` (kind `init` blob); nodes as above (`b` with `workers=1`). Assert B's cache holds the `teacher` blob of `main` with `meta.kind == "teacher"` and the teacher's architecture, and the `init` blob is listed in B's `Welcome.fixed` and fetched only by a node that runs `main`'s learner (B does not: `init` is not in B's cache; A's is checked through `a`'s cache reader) — the spec's "или назначенному лёрнеру" rule.
Both tests: 2 loaded processes (A's learner, B's worker); they end with `hub.stop() == 0` (exit code of `request_stop`). They rely on T2.5's interim collect rule ("until T3.2 collecting seats only for agents whose learner runs on the requesting node") granting node b **non-collecting** lineups (seats of `main@latest` with `collect=False`). If T2.5 grants a node that cannot collect no cells at all, change that interim rule in `_on_lineup_request` to "strip `collect` from seats of agents without a learner on the requesting node" (T3.2 replaces the rule anyway, Step 7 of T3.2).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_version_source.py tests/integration/test_sp5_node_fetch.py tests/integration/test_sp5_weights_cluster.py -q -rw` → all pass (about 60 s together; if a test takes more than 20 s alone, mark it `slow` and say so in the task report).

Full fast suite + ruff, then commit and push part (a):
```bash
git add src/colosseum/metrics/counters.py src/colosseum/node/fetch.py src/colosseum/node/dataport.py \
        src/colosseum/net/proto src/colosseum/node/weights.py src/colosseum/node/agent.py src/colosseum/hub \
        tests/cluster_kit.py tests/fault_proxy.py tests/unit/test_sp5_fault_proxy.py tests/unit/test_sp5_counter_book.py tests/unit/test_sp5_fetch_planner.py \
        tests/unit/test_sp5_data_port.py tests/unit/test_sp5_version_source.py tests/integration/test_sp5_node_fetch.py \
        tests/integration/test_sp5_weights_cluster.py pyproject.toml
git commit -m "feat: latest weights across machines (announcements, data port GetLatest, one fetch per machine)"
git push
```

#### Part (b)

- [ ] **Step 10: Write the failing hold tests (pure, checkpoint manager, blob store)**

Create `tests/unit/test_sp5_snapshot_holds.py`:
```python
"""SnapshotHolds (SP5 T3.1, spec 4.2): an evicted snapshot stays downloadable while a granted lineup names it, and
at most EVICTED_HOLD_FACTOR x match_refresh_interval_sec after its eviction."""
from __future__ import annotations

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.state import EVICTED_HOLD_FACTOR, CellKey, SnapshotHolds, snapshot_refs

A, B = CellKey(1, 0, 0), CellKey(1, 0, 1)


def _lineup(*networks):
    return Lineup("2p", [SeatAssignment("main", n, collect=n == "latest") for n in networks])


def test_snapshot_refs_lists_only_snapshot_seats():
    assert snapshot_refs(_lineup("latest", "ckpt_v40")) == {("main", "ckpt_v40")}
    assert snapshot_refs(Lineup("2p", [SeatAssignment("random", "fixed", collect=False)])) == frozenset()


def test_references_follow_the_newest_grant_of_each_cell():
    h = SnapshotHolds(30.0)
    h.grant(A, {("main", "ckpt_v40")})
    h.grant(B, {("main", "ckpt_v40")})
    assert h.referenced("main", "ckpt_v40")
    h.grant(A, {("main", "ckpt_v60")})                            # a new grant replaces A's references
    assert h.referenced("main", "ckpt_v40") and h.referenced("main", "ckpt_v60")
    h.drop_cell(B)
    assert not h.referenced("main", "ckpt_v40")
    h.drop_node(1)
    assert not h.referenced("main", "ckpt_v60")


def test_a_hold_ends_when_no_lineup_names_the_snapshot_or_at_its_deadline():
    h = SnapshotHolds(10.0)
    assert h.max_hold_sec == EVICTED_HOLD_FACTOR * 10.0
    h.grant(A, {("main", "ckpt_v40")})
    h.grant(B, {("main", "ckpt_v50")})
    h.hold("main", "ckpt_v40", now=100.0)
    h.hold("main", "ckpt_v50", now=100.0)
    assert h.due(now=101.0) == []
    h.grant(A, ())                                                # A's next grant no longer names v40
    assert h.due(now=101.0) == [("main", "ckpt_v40")]
    assert h.is_held("main", "ckpt_v50") and not h.is_held("main", "ckpt_v40")
    assert h.due(now=119.9) == []
    assert h.due(now=120.0) == [("main", "ckpt_v50")]             # still named, but the deadline passed
    assert h.held() == []


def test_without_a_match_refresh_nothing_is_held():
    h = SnapshotHolds(0.0)                                        # match_refresh_interval_sec: 0
    h.grant(A, {("main", "ckpt_v40")})
    assert h.max_hold_sec == 0.0 and not h.referenced("main", "ckpt_v40")
```

Create `tests/unit/test_sp5_checkpoint_hold.py`:
```python
"""The checkpoint manager's eviction hold (SP5 T3.1, spec 4.2): a snapshot evicted while the hub says a granted
lineup names it is kept (hard links) under checkpoints/.evicted/<agent>/<id>/ until the hub releases it; the hold
never fails a save and is invisible to the pool, rescans, run-dir listings and imports."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from colosseum.coordinator.checkpoint_manager import EVICTED_DIR, CheckpointManager, list_snapshots

STATE = {"w": np.ones(3, np.float32)}


def _manager(tmp_path, **kw):
    return CheckpointManager(tmp_path / "run" / "checkpoints", keep_last=1, **kw)


def test_an_evicted_snapshot_named_by_a_lineup_is_kept_under_evicted(tmp_path):
    evicted = []
    m = _manager(tmp_path, on_evict=lambda a, c: evicted.append((c, m.held_dir(a, c).is_dir())))
    m.set_eviction_hold(lambda agent, ckpt: ckpt == "ckpt_v10")
    m.save("main", 10, STATE)
    m.save("main", 20, STATE)
    held = m.held_dir("main", "ckpt_v10")
    assert held == m.base_dir / EVICTED_DIR / "main" / "ckpt_v10"
    assert evicted == [("ckpt_v10", True)]                    # the copy exists when the listeners run
    assert sorted(p.name for p in held.iterdir()) == ["meta.json", "model.pt"]
    assert [c.checkpoint_id for c in m.list_checkpoints("main")] == ["ckpt_v20"]
    assert list(list_snapshots(tmp_path / "run")) == ["main"]  # .evicted is not an agent
    assert not (m.base_dir / "main" / "ckpt_v10").exists()
    m.save("main", 30, STATE)                                 # v20 is not named by a lineup: deleted
    assert not m.held_dir("main", "ckpt_v20").exists()
    m.release_held("main", "ckpt_v10")
    assert not held.exists()
    assert CheckpointManager(m.base_dir, keep_last=1).agents == ["main"]   # a rescan ignores .evicted


def test_a_failing_hold_check_never_fails_the_save(tmp_path, caplog):
    m = _manager(tmp_path)
    m.set_eviction_hold(lambda a, c: 1 / 0)
    m.save("main", 10, STATE)
    with caplog.at_level(logging.WARNING):
        m.save("main", 20, STATE)
    assert not m.held_dir("main", "ckpt_v10").exists()
    assert "not held" in caplog.text


def test_clear_held_removes_every_leftover(tmp_path):
    m = _manager(tmp_path)
    m.set_eviction_hold(lambda a, c: True)
    for v in (10, 20, 30):
        m.save("main", v, STATE)
    assert m.held_dir("main", "ckpt_v20").is_dir()
    m.clear_held()
    assert not (m.base_dir / EVICTED_DIR).exists()


def test_a_run_dir_import_skips_the_evicted_dir(tmp_path):
    parent = _manager(tmp_path / "parent")
    parent.set_eviction_hold(lambda a, c: True)
    for v in (10, 20):
        parent.save("main", v, STATE)
    child = CheckpointManager(tmp_path / "child" / "checkpoints", keep_last=5)
    assert child.import_snapshots(parent.base_dir, "main") == ["ckpt_v20"]


def test_bad_ids_are_refused(tmp_path):
    with pytest.raises(Exception):
        _manager(tmp_path).held_dir("main", "../x")
```

Create `tests/unit/test_sp5_blob_store_hold.py` (T2.3's `BlobStore` on a run dir with `checkpoints/main/ckpt_v10/` written by `CheckpointManager.save` from a tic-tac-toe model's numpy state dict and `meta_extra` with `roles` / `role_signature` / `networks` as T2.5 writes them):
- `test_a_held_snapshot_is_served_from_the_evicted_dir`: evict v10 with the hold on (`keep_last=1`, save v20); `BlobStore(run_dir).frame_path("snapshot", "main", "ckpt_v10")` returns a path whose frame decodes (`decode_weights`) to the same arrays as v10's `model.pt` with `meta.kind == "snapshot"` and `meta.checkpoint_id == "ckpt_v10"`.
- `test_a_released_snapshot_is_not_found`: `release_held` + `BlobStore.forget("snapshot", "main", "ckpt_v10")` → the converted frame file is gone and `frame_path(...)` returns None (the Download servicer answers `NOT_FOUND`; the node then counts `lineup_fetch_failed`).
- `test_forget_during_a_conversion_leaves_no_frame`: monkeypatch the conversion to call `forget(...)` from inside (simulating the main loop releasing while a servicer converts) → after `frame_path` returns, no frame file of v10 exists in `runs/<name>/blobs/` and the call returned None.

Run the three files: `ImportError` (`EVICTED_DIR`, `SnapshotHolds`) / `AttributeError` (`set_eviction_hold`, `forget`).

- [ ] **Step 11: `SnapshotHolds` and `snapshot_refs` (`src/colosseum/hub/state.py`)**

Add after `CellTable`:
```python
def snapshot_refs(lineup: Lineup) -> frozenset[tuple[str, str]]:
    """``(agent_id, checkpoint_id)`` of every snapshot seat (``ckpt_v<N>``) of ``lineup``."""
    return frozenset((s.agent_id, s.network_id) for s in lineup.seats if snapshot_version(s.network_id) is not None)


class SnapshotHolds:
    """Spec 4.2: an evicted snapshot stays downloadable while a granted lineup names it, and at most
    ``EVICTED_HOLD_FACTOR x rollout.match_refresh_interval_sec`` after its eviction (pure; hub clock).

    A cell's references are those of its newest grant (a new grant replaces them; ``drop_cell`` / ``drop_node``
    when the cell or its node goes). ``referenced`` answers the checkpoint manager's question at eviction time
    ("keep a copy?"); ``hold`` starts the hold of a snapshot that was copied; ``due`` returns (and forgets) the
    holds to release: no longer referenced, or past their deadline. With ``match_refresh_interval_sec: 0`` nothing
    is held (the hold bound is 0; a node that could not fetch re-requests, spec 4.2)."""

    def __init__(self, refresh_sec: float, *, factor: float = EVICTED_HOLD_FACTOR) -> None:
        self.max_hold_sec = max(0.0, float(factor) * float(refresh_sec))
        self._cells: dict[CellKey, frozenset[tuple[str, str]]] = {}
        self._counts: collections.Counter = collections.Counter()
        self._held: dict[tuple[str, str], float] = {}          # (agent, ckpt) -> deadline

    def grant(self, cell: CellKey, refs: Iterable[tuple[str, str]]) -> None:
        new = frozenset(refs)
        old = self._cells.get(cell, frozenset())
        if new == old:
            return
        self._counts.subtract(old)
        self._counts.update(new)
        self._counts = +self._counts                              # drop zero entries
        if new:
            self._cells[cell] = new
        else:
            self._cells.pop(cell, None)

    def drop_cell(self, cell: CellKey) -> None:
        self.grant(cell, ())

    def drop_node(self, node: int) -> None:
        for cell in [c for c in self._cells if c.node == node]:
            self.drop_cell(cell)

    def referenced(self, agent_id: str, checkpoint_id: str) -> bool:
        """Does a granted lineup name this snapshot (and is a hold possible at all: ``max_hold_sec > 0``)?"""
        return self.max_hold_sec > 0 and self._counts[(agent_id, checkpoint_id)] > 0

    def hold(self, agent_id: str, checkpoint_id: str, now: float) -> None:
        self._held[(agent_id, checkpoint_id)] = now + self.max_hold_sec

    def is_held(self, agent_id: str, checkpoint_id: str) -> bool:
        return (agent_id, checkpoint_id) in self._held

    def held(self) -> list[tuple[str, str]]:
        return sorted(self._held)

    def due(self, now: float) -> list[tuple[str, str]]:
        out = [key for key, deadline in self._held.items() if now >= deadline or self._counts[key] <= 0]
        for key in out:
            del self._held[key]
        return sorted(out)
```
Imports: `collections`, `from collections.abc import Iterable`, `from colosseum.arena.entities import snapshot_version`, `from colosseum.core.types import Lineup` (whatever `hub.state` does not import yet).

- [ ] **Step 12: The checkpoint manager hold (`src/colosseum/coordinator/checkpoint_manager.py`)**

1. Module docstring, after the eviction paragraph: "SP5 (spec 4.2): with an eviction hold set (``set_eviction_hold``) an evicted snapshot the hold callback names is first hard-linked (``model.pt``, ``meta.json``; copied when links fail) into ``base_dir/.evicted/<agent_id>/<checkpoint_id>/``, so the hub can still serve it to nodes whose granted lineups name it; the hub releases it (``release_held``) when no granted lineup names it any more, at the latest ``2 x match_refresh_interval_sec`` later, and clears the dir at start (``clear_held``). ``.evicted`` starts with a dot: scans, ``list_snapshots`` and imports skip it."
2. Constant `EVICTED_DIR = ".evicted"`; in `__init__`, `self._hold: Callable[[str, str], bool] | None = None`.
3. New methods:
```python
    def set_eviction_hold(self, hold: Callable[[str, str], bool] | None) -> None:
        """``hold(agent_id, checkpoint_id)``: keep a copy of this snapshot when it is evicted (main loop; the hub's
        ``SnapshotHolds.referenced``)."""
        self._hold = hold

    def held_dir(self, agent_id: str, checkpoint_id: str) -> Path:
        return (self._base_dir / EVICTED_DIR / check_agent_id(agent_id)
                / check_path_component(checkpoint_id, "checkpoint id"))

    def release_held(self, agent_id: str, checkpoint_id: str) -> None:
        shutil.rmtree(self.held_dir(agent_id, checkpoint_id), ignore_errors=True)

    def clear_held(self) -> None:
        """Remove every held copy (hub start: no granted lineup survives a hub restart)."""
        shutil.rmtree(self._base_dir / EVICTED_DIR, ignore_errors=True)

    def _wants_hold(self, agent_id: str, checkpoint_id: str) -> bool:
        if self._hold is None:
            return False
        try:
            return bool(self._hold(agent_id, checkpoint_id))
        except Exception:  # noqa: BLE001 - a hold check never fails a save or an eviction
            logger.warning(f"Eviction hold check failed for {checkpoint_id} of {agent_id}; not held", exc_info=True)
            return False

    def _keep_copy(self, agent_id: str, checkpoint_id: str, src: Path) -> None:
        target = self.held_dir(agent_id, checkpoint_id)
        tmp = target.parent / f"{_TMP_PREFIX}{checkpoint_id}-{uuid.uuid4().hex[:8]}"
        try:
            tmp.mkdir(parents=True)
            for name in (MODEL_FILE, META_FILE):
                link_or_copy(src / name, tmp / name)
            shutil.rmtree(target, ignore_errors=True)
            os.replace(tmp, target)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            logger.warning(f"Could not keep evicted snapshot {checkpoint_id} of {agent_id} for granted lineups: {e}")
```
4. `_evict`: between the rename to `.tmp-evict-...` and the `rmtree`, insert `if self._wants_hold(agent_id, checkpoint_id): self._keep_copy(agent_id, checkpoint_id, doomed)`. The callback order stays: the index is updated, the copy exists, then `on_evict` runs (the hub's listener sees `held_dir(...).is_dir()`).

Run `tests/unit/test_sp5_checkpoint_hold.py` and `tests/unit/test_sp5_snapshot_holds.py` → pass; the existing checkpoint tests (`tests/unit/test_sp2_checkpoint*.py`, `test_sp4_retention*.py`, `test_sp4_snapshot_store.py`) stay green.

- [ ] **Step 13: `BlobStore` (`src/colosseum/hub/server.py`)**

1. `frame_path("snapshot", agent_id, ident)`: the source dir is `checkpoints/<agent>/<ident>/` when its `model.pt` exists, else `checkpoints/.evicted/<agent>/<ident>/`, else None. The converted frame stays where T2.3 puts it (`runs/<name>/blobs/snapshot-<agent>-<ident>.frame`).
2. `forget(kind, agent_id, ident)` (thread-safe; called by the main loop): adds `(kind, agent_id, ident)` to a `_forgotten` set and deletes the converted frame file if it exists. A conversion (servicer thread) holds the store's lock only around its final `os.replace(tmp, final)`: when the key is in `_forgotten` it deletes `tmp` and returns None instead. A later `frame_path` of a forgotten key that finds its source again (a re-saved id, e.g. after a resume) removes the key from `_forgotten` and converts as usual.
Run `tests/unit/test_sp5_blob_store_hold.py` → pass.

- [ ] **Step 14: Hub wiring of the hold and `lineup_fetch_failed` (`src/colosseum/hub/hub.py`)**

1. Start (`launch`, before the server accepts sessions): `self._holds = SnapshotHolds(cfg.rollout.match_refresh_interval_sec)`; `cm = coordinator.checkpoint_manager`; `cm.clear_held()`; `cm.set_eviction_hold(self._holds.referenced)`.
2. `_on_lineup_request(...)`: after a grant is made for a cell, `self._holds.grant(cell_key, snapshot_refs(lineup))` (a non-collecting or fixed-only lineup has no refs; a cell granted nothing keeps its previous refs until its next grant).
3. Wherever cells disappear (`CellTable.drop_node` on node loss or removal; a cell reported with a new incarnation, which gets a new epoch): `self._holds.drop_node(number)` / `self._holds.drop_cell(key)`.
4. `_on_snapshot_evicted(agent_id, checkpoint_id)` (the existing evict listener, which T2.5 also uses for the journal and the eviction announcements): `if cm.held_dir(agent_id, checkpoint_id).is_dir(): self._holds.hold(agent_id, checkpoint_id, now); self._book.add("snapshot_holds", 1)` else `self._blobs.forget("snapshot", agent_id, checkpoint_id)`. The eviction announcement to the nodes is sent either way (nodes drop the snapshot from their caches when no lineup uses it, T2.4).
5. Main loop, every pass (cheap): `for agent_id, ckpt in self._holds.due(now): cm.release_held(agent_id, ckpt); self._blobs.forget("snapshot", agent_id, ckpt)`. At stop (`_shutdown`, after the nodes are stopped): release every held snapshot (`clear_held`) and forget their blobs.
6. `lineup_fetch_failed` reaches `hub_state.json` through the book (Step 8); T7.1 prints it.

- [ ] **Step 15: The hold and two-node tests**

Create `tests/integration/test_sp5_snapshot_hold.py`:
- `test_snapshots_evicted_while_referenced_stay_downloadable`: `cluster_config({"checkpoint.interval": 2, "checkpoint.keep_last": 1, "checkpoint.keep_every": 0, "matchmaking.opponents.latest": 0.0, "matchmaking.opponents.snapshots": 1.0, "matchmaking.opponents.anchors": 0.0, "rollout.match_refresh_interval_sec": 0.5, "training.total_timesteps": 10**9})`; one node `a` with `cpu_learners=1, workers=1` (2 loaded: its learner and its worker; the node fetches its snapshot opponents from the hub like any other node). The learner saves a snapshot every 2 versions and keeps only the newest, so snapshots are evicted all the time while granted lineups name them. Wait until `read_hub_state(hub)["counters"]["book"]["totals"]["snapshot_holds"] >= 3` (timeout 90 s). Assert: `a.agent.counters()["lineup_fetch_failed"] == 0` (every grant's snapshot could be downloaded: from the store or from `.evicted`); after `hub.stop()`, `(hub.root / "checkpoints" / ".evicted").exists()` is False; the journal has results whose seats have `network_id` starting with `ckpt_v` (snapshot opponents were played). If the test is longer than ~20 s, mark it slow (it must stay in the suite).

Create `tests/integration/test_sp5_weights_two_nodes_slow.py` (`@pytest.mark.slow`; 3 loaded processes: A's learner, A's worker, B's worker):
- `test_the_second_node_follows_the_learners_versions_with_one_download_each`: `cluster_config({"rollout.weight_sync_interval_sec": 1.0, "training.total_timesteps": 60000})`; nodes `a` (`cpu_learners=1, workers=1`: A collects for its own learner, which trains) and `b` (`workers=1, no_learners=True`: plays `main@latest` without collecting until T3.2); run until the hub's budget stops it (`hub.thread.join(240)`; exit code 0). Assert: `history = b.agent.fetch_history()["main"]` is strictly increasing (no version downloaded twice) with at least 3 entries; `b.agent.counters()["weight_fetches"] == len(history)` and `["weight_fetch_redundant"] == 0` (criterion 4: one network download per (machine, agent, version), counted where it happens); node a fetched nothing of its own agent (`a.agent.counters().get("weight_fetches", 0) == 0`); the hub book's `weight_fetches` total equals the sum over both nodes; node b's journaled results have at least 3 distinct `version_end` values of their `main` seats (B played with new weights). The same rule with several workers on one node is `test_sp5_node_fetch.py::test_one_download_per_version_whatever_the_number_of_workers` (2 workers, fast).

- [ ] **Step 16: Run, full fast suite, commit and push part (b)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_snapshot_holds.py tests/unit/test_sp5_checkpoint_hold.py tests/unit/test_sp5_blob_store_hold.py tests/integration/test_sp5_snapshot_hold.py -q -rw` → pass; `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_weights_two_nodes_slow.py -m slow -q -rw` (alone, quiet machine) → pass. Full fast suite + ruff. Commit and push:
```bash
git add src/colosseum/hub src/colosseum/coordinator/checkpoint_manager.py tests/unit/test_sp5_snapshot_holds.py \
        tests/unit/test_sp5_checkpoint_hold.py tests/unit/test_sp5_blob_store_hold.py \
        tests/integration/test_sp5_snapshot_hold.py tests/integration/test_sp5_weights_two_nodes_slow.py
git commit -m "feat: evicted snapshots held for granted lineups; lineup_fetch_failed and fetch counters in the hub"
git push
```

---

### Task T3.2: Chunk routing between machines and the hub relay (chunks and weights)

Spec 4.7 across machines, 4.8, 3.1 (the local relay to the hub machine's learners), 7.1 ("разрыв сети при живой машине" for the data plane). After T3.1 every node has a data port and fetches remote latest weights directly. This task makes workers collect for learners on other machines: a worker hands frames for a remote learner to its node over its route connection (T2.2 `WorkerIO` already does that); the node's `ChunkRouter` reads the frame header only (`peek_meta`), puts the frame on the target node's `TargetSender` queue (blocking: backpressure to the worker), and the sender streams `ChunkFrame` messages to the target's data port (`NodeData.Chunks`, mutual TLS) when one of its announced addresses answers within `distributed.connect_timeout_sec`, else through the hub (`Hub.RelaySend` → `RelayHub` pair buffer → the target's `Hub.RelayInbox` with credits); the receiving node hands each frame to the learner's chunk connection (`LocalDelivery`, blocking). Nothing in between decodes a frame. Weights of a learner whose data port is unreachable come through the hub too (`Download(kind="latest")` → one pull per version down the learner node's `RelayInbox` → `Upload(kind="relay_latest")`). The hub grants collecting seats for an agent to any node from which that agent's learner is reachable. Every lost frame is counted by reason.

**Split point:** dispatch (a) Steps 1–8 (the router, `DataServer.Chunks`, the node routing section, the collect rule, `Heartbeat.paths`, the direct-path tests), then (b) Steps 9–15 (`hub.relay`, the relay servicers, relayed weights, the node's relay receiver and pull uploads, `relay: false`, the relay cluster tests). Each dispatch ends with the full fast suite, a commit and a push. Until (b) lands, a sender without a direct path reports "down" (the relay callable is None).

**Files:**
- Create: `src/colosseum/node/router.py` (`ChunkRouter`, `RouteTable`, `RemoteRoute`, `TargetSender`, `RouteReader`, `LocalDelivery`, `RelayReceiver`, `Counters`, constants)
- Create: `src/colosseum/hub/relay.py` (`RelayHub`, `InboxHandle`, `LatestRelay`, `RelayedVersion`, constants)
- Modify: `src/colosseum/node/dataport.py` (`DataServer.Chunks`; `fetch_latest_relay`)
- Modify: `src/colosseum/hub/server.py` (relay servicers `RelaySend` / `RelayInbox`, `Download(kind="latest")`, `Upload(kind="relay_latest")`; `build_server(..., relay=, latest=)`)
- Modify: `src/colosseum/hub/state.py` (`collect_allowed`, `strip_collect`)
- Modify: `src/colosseum/hub/hub.py` (the collect rule in `_on_lineup_request`, `Heartbeat.paths`, `RelayHub.set_online`, `LatestRelay.publish_versions`, relay counters into the book)
- Modify: `src/colosseum/node/agent.py` (the routing section: router, routes, `LearnerRoute`s, `LocalDelivery`, `RelayReceiver`, pull uploads, relay fallback for weights, `Heartbeat.paths`)
- Modify: `src/colosseum/net/proto/colosseum_v1.proto` + regenerated code (`PathState`, `Heartbeat.paths`)
- Test: `tests/unit/test_sp5_chunk_router.py`, `tests/unit/test_sp5_relay_hub.py`, `tests/unit/test_sp5_collect_rule.py`, `tests/unit/test_sp5_data_port_chunks.py`, `tests/integration/test_sp5_relay_servicers.py`, `tests/integration/test_sp5_chunks_cluster.py`, `tests/integration/test_sp5_chunks_cluster_slow.py`; extended: `tests/integration/test_sp5_node_fetch.py`

**Interfaces:**
- Consumes: T3.1's `DataServer`, `probe_direct`, `read_latest_reply`, `check_fetched`, `FetchError`, `FetchPlanner`, `CounterBook`, `cluster_kit`, `fault_proxy`, `VersionInfo` with `addresses` / `relay_only`; `peek_meta`, `FrameError` (T1.1); `ChunkFrame`, `ChunkSummary`, `RelayCredit`, `RelayItem`, `LatestRequest`, `UploadPart` / `UploadHeader` / `UploadReply`, `BlobRequest`, `BlobPart` (T1.2); `send_parts`, `PART_BYTES`, the auth metadata of R1 (`MD_NODE_ID`, `MD_NODE_SECRET`) and `AuthInterceptor` (T1.3, T2.3); `LearnerRoute`, `WorkerIO` route connections, the node's chunk connection to each local learner (T2.2, T2.4); `OwnerRotation.assign(collecting=...)` (T2.1); `Hub._on_lineup_request`, `_on_heartbeat`, `_on_session_opened` / `_closed`, node loss (T2.5).
- Produces (contract additions): the `router` and `relay` names above; `DataServer.Chunks`; `fetch_latest_relay(channel, agent_id, since_version, *, max_message_mb, metadata) -> FetchedLatest | None`; `collect_allowed`, `strip_collect`; `build_server(..., relay: RelayHub | None = None, latest: LatestRelay | None = None)`; proto `PathState`, `Heartbeat.paths`; counter names `relay_bytes`, `relay_frames`, `chunks_sent.direct|relay`, `chunks_delivered`, `relay_frames_received`, `dropped_chunks.{disconnect,no_learner,invalid,overflow}`.

#### Part (a): the direct path

- [ ] **Step 1: Write the failing router tests**

Create `tests/unit/test_sp5_chunk_router.py`:
```python
"""The node chunk router without gRPC (SP5 T3.2, spec 4.7, 4.8): frames are routed by their header, unknown agents
and malformed frames are counted, a slow target holds the worker back, a down path drops and counts, paths are
reported."""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np

from colosseum.net.frame import encode_frame
from colosseum.node.router import ChunkRouter, RemoteRoute


def _frame(agent, gen=1, size=100):
    """A frame whose header looks like a chunk frame's (``peek_meta`` is all the router reads)."""
    meta = {"agent_id": agent, "generation": gen, "policy_version": 0, "node": 1, "worker": 0}
    return encode_frame({"x": np.zeros(size, np.uint8)}, kind="chunk", meta=meta)


class FakeRelay:
    def __init__(self, delay=0.0):
        self.got = []
        self.delay = delay

    def __call__(self):
        def call(frames):
            for f in frames:
                time.sleep(self.delay)
                self.got.append(f)
            return None
        return call


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_frames_go_to_their_agents_target_and_unknown_agents_are_counted():
    relay = FakeRelay()
    paths = []
    router = ChunkRouter(make_frame=lambda a, g, f, t: (a, g, t, len(f)), make_channel=None, direct_stub=None,
                         relay_call=relay, connect_timeout_sec=0.5, on_path=lambda t, p: paths.append((t, p)))
    router.set_routes({"m": RemoteRoute(node=2, generation=1, addresses=(), relay_only=True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    for _ in range(5):
        worker.send_bytes(_frame("m"))
    worker.send_bytes(_frame("ghost"))
    worker.send_bytes(b"garbage")
    assert _wait(lambda: len(relay.got) == 5 and router.counters().get("dropped_chunks.invalid") == 1)
    assert relay.got[0][:3] == ("m", 1, 2)
    assert router.counters()["dropped_chunks.no_learner"] == 1
    assert paths == [(2, "relay")] and router.paths() == {2: "relay"}
    router.stop()


def test_a_slow_target_blocks_the_worker_instead_of_dropping():
    relay = FakeRelay(delay=0.05)
    router = ChunkRouter(make_frame=lambda a, g, f, t: a, make_channel=None, direct_stub=None, relay_call=relay,
                         connect_timeout_sec=0.5)
    router.set_routes({"m": RemoteRoute(2, 1, (), True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    sent = []

    def produce():
        for _ in range(60):
            worker.send_bytes(_frame("m", size=200_000))     # larger than the pipe buffer
            sent.append(1)
    t = threading.Thread(target=produce, daemon=True)
    t.start()
    time.sleep(0.5)
    assert len(sent) < 60                      # the worker is held back (the sender queue is full)
    assert _wait(lambda: len(relay.got) == 60, timeout=15)
    assert "dropped_chunks.disconnect" not in router.counters()
    router.stop()


def test_without_relay_and_without_a_direct_path_frames_are_dropped_and_the_path_is_down():
    paths = []
    router = ChunkRouter(make_frame=lambda a, g, f, t: a, make_channel=None, direct_stub=None, relay_call=None,
                         connect_timeout_sec=0.2, on_path=lambda t, p: paths.append(p))
    router.set_routes({"m": RemoteRoute(2, 1, (), True)})
    worker, node_side = mp.Pipe()
    router.add_worker(node_side, worker=0)
    for _ in range(3):
        worker.send_bytes(_frame("m"))
    assert _wait(lambda: router.counters().get("dropped_chunks.disconnect", 0) == 3)
    assert paths[0] == "down"
    router.stop()
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_chunk_router.py -q` → `ModuleNotFoundError: No module named 'colosseum.node.router'`.

- [ ] **Step 2: Write `src/colosseum/node/router.py`**

```python
"""Chunk routing between machines (SP5 spec 4.7, 4.8; T3.2): node side.

- ``RouteTable`` (thread-safe snapshot, written by the node main loop): agent -> ``RemoteRoute`` (target node,
  generation, data addresses, relay-only) for learners on other nodes.
- ``RouteReader`` threads (one per worker route connection): ``recv_bytes`` a frame, ``peek_meta`` for its agent,
  put it on the target's ``TargetSender`` queue (blocking: the worker's ``send_bytes`` then blocks, spec 4.7
  backpressure); no route -> dropped ``no_learner``.
- ``TargetSender`` threads (one per target node): path choice (direct ``NodeData.Chunks`` when one announced data
  address answers within ``connect_timeout_sec``, else ``Hub.RelaySend`` when ``distributed.relay``, else "down");
  frames are forwarded undecoded in ``ChunkFrame`` messages; a stream is rotated every ``STREAM_ROTATE_FRAMES``
  frames or ``STREAM_ROTATE_SEC`` seconds so its ``ChunkSummary`` confirms what arrived; when a stream fails, the
  frames handed to it since its start are counted ``disconnect`` (an upper bound: some may have arrived) and the
  path is chosen again (a failed direct path is retried after ``DIRECT_RETRY_SEC``).
- ``LocalDelivery`` (receiving side, shared by the data-port ``Chunks`` servicer and ``RelayReceiver``): the frame
  goes to the local learner's chunk connection with ``send_bytes`` under that connection's lock (blocking while the
  learner does not read); no learner -> ``no_learner``, a broken connection -> ``disconnect``.
- ``RelayReceiver``: the node's ``Hub.RelayInbox`` stream: ``RELAY_INITIAL_CREDITS`` credits, then one per frame
  delivered; pull items (weights of a relay-only learner) go to the ``on_pull`` callback (the node uploads its cache's
  newest frame with ``Hub.Upload(kind="relay_latest")``).
Counters are cumulative for the node process (``counters()``; the node sends them in ``Heartbeat.counters``).
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import grpc

from colosseum.net.frame import FrameError, peek_meta
from colosseum.node.dataport import probe_direct

logger = logging.getLogger(__name__)

SENDER_QUEUE_FRAMES = 32
STREAM_ROTATE_FRAMES = 256
STREAM_ROTATE_SEC = 2.0
DIRECT_RETRY_SEC = 30.0
RETRY_MIN_SEC, RETRY_MAX_SEC = 0.5, 10.0
_POLL_SEC = 0.1
RELAY_INITIAL_CREDITS = 16


class Counters:
    """Thread-safe cumulative counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: collections.Counter = collections.Counter()

    def add(self, name: str, n: int = 1) -> None:
        if n:
            with self._lock:
                self._c[name] += n

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._c)


@dataclass(frozen=True)
class RemoteRoute:
    node: int
    generation: int
    addresses: tuple[str, ...]
    relay_only: bool


class RouteTable:
    """agent -> RemoteRoute for learners on other nodes (immutable snapshots swapped under a lock)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._remote: dict[str, RemoteRoute] = {}

    def set_remote(self, routes: dict[str, RemoteRoute]) -> None:
        with self._lock:
            self._remote = dict(routes)

    def remote(self, agent_id: str) -> RemoteRoute | None:
        with self._lock:
            return self._remote.get(agent_id)

    def targets(self) -> dict[int, RemoteRoute]:
        with self._lock:
            return {r.node: r for r in self._remote.values()}


@dataclass(eq=False)
class _Stream:
    lock: threading.Lock = field(default_factory=threading.Lock)
    n: int = 0
    closed: bool = False


class TargetSender(threading.Thread):
    """Frames for one target node (module docstring). ``direct_stub(channel)`` and ``relay_call()`` return a
    callable ``call(request_iterator) -> ChunkSummary`` (``NodeData.Chunks`` / ``Hub.RelaySend``, a client-streaming
    RPC; the relay one carries the node's auth metadata); ``relay_call`` None: ``distributed.relay: false``.
    ``on_path(target, path)`` is called from this thread when the path changes ("direct", "relay", "down",
    "none"); the node posts it into its inbox."""

    def __init__(self, target: int, *, route: Callable[[], RemoteRoute | None], make_frame: Callable[..., Any],
                 make_channel: Callable[[str], grpc.Channel], direct_stub: Callable[[grpc.Channel], Callable],
                 relay_call: Callable[[], Callable] | None, connect_timeout_sec: float, counters: Counters,
                 on_path: Callable[[int, str], None] = lambda t, p: None) -> None:
        super().__init__(name=f"chunks-to-n{target}", daemon=True)
        self.target = target
        self.queue: queue.Queue = queue.Queue(maxsize=SENDER_QUEUE_FRAMES)
        self._route = route
        self._make_frame = make_frame
        self._make_channel = make_channel
        self._direct_stub = direct_stub
        self._relay_call = relay_call
        self._timeout = float(connect_timeout_sec)
        self._counters = counters
        self._on_path = on_path
        self._halt = threading.Event()
        self._channel: grpc.Channel | None = None
        self._direct_failed_at = -float("inf")
        self.path = "none"

    def stop(self) -> None:
        self._halt.set()

    def put(self, item: tuple, should_stop: Callable[[], bool]) -> bool:
        """Blocking put (backpressure); False when stopping."""
        while not should_stop() and not self._halt.is_set():
            try:
                self.queue.put(item, timeout=_POLL_SEC)
                return True
            except queue.Full:
                continue
        return False

    def _choose(self) -> tuple[str, Callable | None]:
        """direct when an announced address answers within ``connect_timeout_sec``; else relay (when on); else
        down. While on the relay, the direct path is probed again only every ``DIRECT_RETRY_SEC``; without a relay
        it is probed on every attempt (``run`` backs off between attempts)."""
        route = self._route()
        if route is None:
            return "none", None
        now = time.monotonic()
        direct_possible = bool(route.addresses) and not route.relay_only
        recently_failed = now - self._direct_failed_at < DIRECT_RETRY_SEC
        if direct_possible and not (recently_failed and self._relay_call is not None):
            found = probe_direct(route.addresses, self._make_channel, self._timeout)
            if found is not None:
                self._channel = found[1]
                return "direct", self._direct_stub(found[1])
            self._direct_failed_at = now
        if self._relay_call is not None:
            return "relay", self._relay_call()
        return "down", None

    def _frames(self, stream: _Stream) -> Iterator[Any]:
        """Request iterator of one stream. gRPC pulls it from its own thread, which may still be inside ``next``
        when the call has already failed: a frame taken after ``stream.closed`` is counted here, every frame
        counted in ``stream.n`` is counted by ``run`` (under ``stream.lock``), so none is lost uncounted."""
        started = time.monotonic()
        while not self._halt.is_set():
            if stream.n >= STREAM_ROTATE_FRAMES or time.monotonic() - started >= STREAM_ROTATE_SEC:
                return
            try:
                item = self.queue.get(timeout=_POLL_SEC)
            except queue.Empty:
                continue
            with stream.lock:
                if stream.closed:
                    self._counters.add("dropped_chunks.disconnect")
                    return
                stream.n += 1
            yield self._make_frame(*item)

    def run(self) -> None:
        backoff = RETRY_MIN_SEC
        while not self._halt.is_set():
            path, call = self._choose()
            if path != self.path:
                self.path = path
                self._on_path(self.target, path)
            if call is None:                       # no route or path down: drop what waits, try again later
                self._drop_waiting("disconnect" if path == "down" else "no_learner")
                self._halt.wait(backoff)
                backoff = min(RETRY_MAX_SEC, backoff * 2)
                continue
            while not self._halt.is_set():
                stream = _Stream()
                try:
                    call(self._frames(stream))
                    with stream.lock:
                        stream.closed = True
                        self._counters.add(f"chunks_sent.{path}", stream.n)
                    backoff = RETRY_MIN_SEC
                except grpc.RpcError as e:
                    with stream.lock:
                        stream.closed = True
                        lost = stream.n
                    self._counters.add("dropped_chunks.disconnect", lost)
                    logger.info(f"chunk stream to node {self.target} ({path}) failed: {e.code()}; "
                                f"{lost} frames counted as lost")
                    if path == "direct":
                        self._direct_failed_at = time.monotonic()
                        if self._channel is not None:
                            self._channel.close()
                            self._channel = None
                    break
                if path == "relay" and time.monotonic() - self._direct_failed_at >= DIRECT_RETRY_SEC:
                    break                          # time to try the direct path again
        self._drop_waiting("disconnect")

    def _drop_waiting(self, reason: str) -> None:
        n = 0
        while True:
            try:
                self.queue.get_nowait()
                n += 1
            except queue.Empty:
                break
        self._counters.add(f"dropped_chunks.{reason}", n)


class LocalDelivery:
    """Frames from other machines to this node's learners (module docstring)."""

    def __init__(self, counters: Counters) -> None:
        self._lock = threading.Lock()
        self._learners: dict[str, tuple[Any, threading.Lock]] = {}
        self._counters = counters

    def set_learner(self, agent_id: str, conn: Any | None) -> None:
        with self._lock:
            old = self._learners.pop(agent_id, None)
            if conn is not None:
                self._learners[agent_id] = (conn, threading.Lock())
        if old is not None and old[0] is not conn:
            try:
                old[0].close()
            except OSError:
                pass

    def count(self, name: str, n: int = 1) -> None:
        self._counters.add(name, n)

    def deliver(self, agent_id: str, frame: bytes) -> str | None:
        with self._lock:
            entry = self._learners.get(agent_id)
        if entry is None:
            self._counters.add("dropped_chunks.no_learner")
            return "no_learner"
        conn, lock = entry
        try:
            with lock:
                conn.send_bytes(frame)
        except (OSError, EOFError, ValueError):
            with self._lock:
                if self._learners.get(agent_id) is entry:
                    del self._learners[agent_id]
            self._counters.add("dropped_chunks.disconnect")
            return "disconnect"
        self._counters.add("chunks_delivered")
        return None


class RelayReceiver(threading.Thread):
    """This node's ``Hub.RelayInbox`` stream (module docstring): frames relayed by the hub go to ``delivery``;
    each delivered (or dropped) frame returns one credit. ``open_inbox(request_iterator)`` returns the response
    iterator of the bidi call; ``on_pull(item)`` handles a weights pull (it must not block: the node hands it to
    its transfer pool). Reconnects with backoff ``RETRY_MIN_SEC`` -> ``RETRY_MAX_SEC``."""

    def __init__(self, *, node_id: str, open_inbox: Callable[[Iterator[Any]], Iterator[Any]],
                 make_credit: Callable[[int, str], Any], delivery: LocalDelivery,
                 chunk_of: Callable[[Any], tuple[str, bytes] | None], on_pull: Callable[[Any], None],
                 counters: Counters) -> None:
        super().__init__(name="relay-inbox", daemon=True)
        self._node_id = node_id
        self._open = open_inbox
        self._make_credit = make_credit
        self._delivery = delivery
        self._chunk_of = chunk_of
        self._on_pull = on_pull
        self._counters = counters
        self._halt = threading.Event()
        self.connected = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def _credits(self, grants: queue.Queue) -> Iterator[Any]:
        yield self._make_credit(RELAY_INITIAL_CREDITS, self._node_id)
        while not self._halt.is_set():
            try:
                n = grants.get(timeout=_POLL_SEC)
            except queue.Empty:
                continue
            if n is None:
                return
            yield self._make_credit(n, self._node_id)

    def run(self) -> None:
        backoff = RETRY_MIN_SEC
        while not self._halt.is_set():
            grants: queue.Queue = queue.Queue()
            responses = self._open(self._credits(grants))
            try:
                for item in responses:
                    self.connected.set()
                    backoff = RETRY_MIN_SEC
                    chunk = self._chunk_of(item)
                    if chunk is None:
                        self._on_pull(item)
                        continue
                    agent_id, frame = chunk
                    self._delivery.deliver(agent_id, frame)       # counts its own drops
                    self._counters.add("relay_frames_received")
                    grants.put(1)
                    if self._halt.is_set():
                        break
            except grpc.RpcError as e:
                logger.info(f"relay inbox stream ended: {e.code()}")
            finally:
                grants.put(None)
                self.connected.clear()
                cancel = getattr(responses, "cancel", None)
                if cancel is not None:
                    cancel()
            self._halt.wait(backoff)
            backoff = min(RETRY_MAX_SEC, backoff * 2)


class RouteReader(threading.Thread):
    """Frames from one worker's route connection (``net.local``) to the target senders (module docstring)."""

    def __init__(self, conn: Any, *, worker: int, routes: RouteTable, sender_for: Callable[[int], TargetSender | None],
                 counters: Counters, stop: threading.Event) -> None:
        super().__init__(name=f"route-w{worker}", daemon=True)
        self._conn = conn
        self._routes = routes
        self._sender_for = sender_for
        self._counters = counters
        self._halt = stop

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                if not self._conn.poll(_POLL_SEC):
                    continue
                frame = self._conn.recv_bytes()
            except (EOFError, OSError):
                break                                   # the worker is gone: its connection only
            try:
                _kind, meta = peek_meta(frame)
                agent_id = str(meta["agent_id"])
                generation = int(meta.get("generation", 0))
            except (FrameError, KeyError, TypeError, ValueError):
                self._counters.add("dropped_chunks.invalid")
                continue
            route = self._routes.remote(agent_id)
            sender = None if route is None else self._sender_for(route.node)
            if sender is None:
                self._counters.add("dropped_chunks.no_learner")
                continue
            if not sender.put((agent_id, generation, frame), self._halt.is_set):
                self._counters.add("dropped_chunks.disconnect")
        try:
            self._conn.close()
        except OSError:
            pass


class ChunkRouter:
    """The node's outbound chunk path (module docstring). The main loop calls ``set_routes`` whenever the version
    registry or this node's learners change, ``add_worker`` for every accepted worker route connection, and reads
    ``paths()`` / ``counters()`` for the heartbeat; ``stop`` at the end. ``make_frame(agent, generation, frame,
    target)`` builds the ``ChunkFrame``; ``direct_stub(channel)`` / ``relay_call()`` return the client-streaming
    callables (``relay_call`` None: ``distributed.relay: false``)."""

    def __init__(self, *, make_frame: Callable[..., Any], make_channel: Callable[[str], grpc.Channel],
                 direct_stub: Callable[[grpc.Channel], Callable], relay_call: Callable[[], Callable] | None,
                 connect_timeout_sec: float, on_path: Callable[[int, str], None] = lambda t, p: None) -> None:
        self.routes = RouteTable()
        self.counters_obj = Counters()
        self.delivery = LocalDelivery(self.counters_obj)
        self._make_frame = make_frame
        self._make_channel = make_channel
        self._direct_stub = direct_stub
        self._relay_call = relay_call
        self._timeout = float(connect_timeout_sec)
        self._on_path = on_path
        self._lock = threading.Lock()
        self._senders: dict[int, TargetSender] = {}
        self._readers: list[RouteReader] = []
        self._halt = threading.Event()

    def set_routes(self, remote: dict[str, RemoteRoute]) -> None:
        self.routes.set_remote(remote)
        targets = {r.node for r in remote.values()}
        with self._lock:
            gone = [t for t in self._senders if t not in targets]
            stopped = [self._senders.pop(t) for t in gone]
        for sender in stopped:
            sender.stop()

    def sender_for(self, target: int) -> TargetSender | None:
        if self._halt.is_set():
            return None
        with self._lock:
            sender = self._senders.get(target)
            if sender is None:
                if target not in self.routes.targets():
                    return None
                sender = TargetSender(
                    target, route=lambda t=target: self.routes.targets().get(t),
                    make_frame=lambda agent, gen, frame, t=target: self._make_frame(agent, gen, frame, t),
                    make_channel=self._make_channel, direct_stub=self._direct_stub, relay_call=self._relay_call,
                    connect_timeout_sec=self._timeout, counters=self.counters_obj, on_path=self._on_path)
                self._senders[target] = sender
                sender.start()
            return sender

    def add_worker(self, conn: Any, worker: int) -> None:
        reader = RouteReader(conn, worker=worker, routes=self.routes, sender_for=self.sender_for,
                             counters=self.counters_obj, stop=self._halt)
        self._readers = [r for r in self._readers if r.is_alive()] + [reader]
        reader.start()

    def paths(self) -> dict[int, str]:
        with self._lock:
            return {t: s.path for t, s in self._senders.items()}

    def counters(self) -> dict[str, int]:
        return self.counters_obj.snapshot()

    def stop(self, timeout: float = 2.0) -> None:
        self._halt.set()
        with self._lock:
            senders = list(self._senders.values())
        for sender in senders:
            sender.stop()
        for thread in [*senders, *self._readers]:
            thread.join(timeout)
```
Notes: the `Thread` subclasses keep their stop event in `_halt` (an attribute named `_stop` shadows `Thread._stop` and breaks `join`). gRPC consumes a request iterator from its own thread: `_Stream` and its lock make "counted in `n` and then lost" and "taken after the stream closed" both counted, never dropped silently (verified with a 1.5 s cut: delivered + counted = 905 for 900 produced). A broken stream counts all its frames as `disconnect`, an upper bound of the true loss (ruling 1): the stream rotation bounds it to `STREAM_ROTATE_FRAMES` + the frames in the gRPC window.

Run the router tests → 3 passed.

- [ ] **Step 3: Write the failing data-port chunk tests**

Create `tests/unit/test_sp5_data_port_chunks.py` (a `DataServer` with `insecure=True`, `delivery=LocalDelivery(Counters())`, `node=2`, started on port 0; "learners" are `multiprocessing.Pipe()` ends registered with `delivery.set_learner(agent, conn)`; a client streams `ChunkFrame`s with `NodeDataStub(channel).Chunks(iter(frames), metadata=[(MD_TOKEN, secret)])`):
- `test_frames_reach_the_local_learner_undecoded`: 10 frames of `encode_frame(...)` bytes for agent `main`, target 2 → the pipe's other end `recv_bytes()` 10 times, each equal to the sent bytes; the summary is `received == 10, dropped == 0`; `counters["chunks_delivered"] == 10`.
- `test_frames_without_a_learner_or_with_a_dead_learner_are_counted`: frames for agent `ghost` (no learner) → `dropped_chunks.no_learner`; frames with `target_node == 3` → `no_learner` too (a frame for another node is never forwarded); close the learner end of `main`'s pipe, send 3 frames → the first send that fails counts `disconnect` and the rest `no_learner` (the learner entry is removed after the failure); the summary's `dropped` equals the sum; the stream never fails (`received == len(frames)`).
- `test_a_slow_learner_backpressures_the_sending_node`: the learner end reads one frame every 20 ms in a thread; a client generator yields 100 frames of 200 KB and records when each is pulled; after 0.5 s fewer than 60 have been pulled (the gRPC window plus the pipe buffer: the sender is held back, nothing is dropped); all 100 arrive in the end and `dropped == 0`.
- `test_chunks_need_the_token_or_a_node_certificate`: without the token metadata the call fails `UNAUTHENTICATED` (TLS + certificates are covered by `test_sp5_data_port.py`).

Run → fail (`UNIMPLEMENTED`).

- [ ] **Step 4: `DataServer.Chunks` (`src/colosseum/node/dataport.py`)**

Replace the T3.1 stub:
```python
    def Chunks(self, request_iterator: Iterable[pb.ChunkFrame], context: grpc.ServicerContext) -> pb.ChunkSummary:  # noqa: N802
        """Frames for this node's learners (spec 4.7): each goes to its learner's chunk connection undecoded
        (``LocalDelivery.deliver``, blocking while the learner does not read: gRPC flow control then holds the
        sender back). A frame for another node or an agent without a learner here is dropped (counted); the stream
        itself never fails because of one frame."""
        self._check_peer(context)
        if self._delivery is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "this node takes no chunks")
        received = dropped = 0
        for frame in request_iterator:
            received += 1
            if frame.target_node != self._node:
                self._delivery.count("dropped_chunks.no_learner")
                dropped += 1
                continue
            if self._delivery.deliver(frame.agent_id, frame.frame) is not None:
                dropped += 1
        return pb.ChunkSummary(received=received, dropped=dropped)
```
`DataServer.__init__` already takes `delivery` (T3.1); the node passes `router.delivery` (Step 6). Run the data-port chunk tests → pass.

- [ ] **Step 5: The collect rule (`src/colosseum/hub/state.py`) — failing tests first**

Create `tests/unit/test_sp5_collect_rule.py`:
```python
"""The hub's collect rule across machines (SP5 T3.2, spec 4.7, 4.8): an agent collects on a node when its learner
runs, on that node or anywhere with the relay on, or on a node with a direct path to it with the relay off."""
from __future__ import annotations

from colosseum.core.types import Lineup, SeatAssignment
from colosseum.hub.state import collect_allowed, strip_collect


def test_collect_needs_a_running_learner_and_a_way_to_it():
    assert collect_allowed(1, None, relay=True, path=None) is False          # no learner (pending, failed)
    assert collect_allowed(1, 1, relay=False, path=None) is True             # the learner is on this node
    assert collect_allowed(1, 2, relay=True, path=None) is True              # direct or relayed
    assert collect_allowed(1, 2, relay=False, path="direct") is True
    assert collect_allowed(1, 2, relay=False, path="down") is False
    assert collect_allowed(1, 2, relay=False, path=None) is False            # not reported yet: not granted


def test_strip_collect_keeps_the_seats_and_turns_off_only_disallowed_collection():
    lineup = Lineup("2p", [SeatAssignment("a", "latest", True, "owner"), SeatAssignment("b", "latest", True, "rivals"),
                           SeatAssignment("b", "ckpt_v4", False, "snapshots")], seed=None, epoch=7)
    out = strip_collect(lineup, {"a"})
    assert [(s.agent_id, s.network_id, s.collect, s.source) for s in out.seats] == [
        ("a", "latest", True, "owner"), ("b", "latest", False, "rivals"), ("b", "ckpt_v4", False, "snapshots")]
    assert out.epoch == 7 and lineup.seats[1].collect is True                # a copy; the input is unchanged
```
Then add to `src/colosseum/hub/state.py`:
```python
def collect_allowed(node: int, learner_node: int | None, *, relay: bool, path: str | None) -> bool:
    """May a seat of an agent collect on ``node`` (spec 4.7: no collecting seats without a learner; 4.8: an
    unreachable learner only through the relay)? ``learner_node``: the node of the agent's running learner (None:
    pending, failed, relocating); ``path``: what ``node`` last reported for the learner's node
    (``Heartbeat.paths``: "direct" / "relay" / "down"; None: not yet)."""
    if learner_node is None:
        return False
    if learner_node == node:
        return True
    return bool(relay) or path == "direct"


def strip_collect(lineup: Lineup, allowed: Collection[str]) -> Lineup:
    """A copy of ``lineup`` whose seats of agents outside ``allowed`` do not collect (they still play)."""
    seats = [s if not s.collect or s.agent_id in allowed else dataclasses.replace(s, collect=False)
             for s in lineup.seats]
    return dataclasses.replace(lineup, seats=seats)
```
Run → 2 passed.

- [ ] **Step 6: Proto, hub wiring of the collect rule and paths (`hub/hub.py`)**

1. Proto: `message PathState { int32 node = 1; string path = 2; }` and `Heartbeat.paths` (`repeated PathState`, the next free number); regenerate; extend the T1.2 round-trip test with one assertion.
2. `_on_heartbeat`: `self._paths[(number, p.node)] = p.path` for every `PathState` (strings outside `{"direct", "relay", "down", "none"}` are dropped, counted `node_messages.invalid` in the book). Node loss drops the node's entries.
3. `_on_lineup_request` (replaces the interim "collect only for a learner on the requesting node" rule of T2.5): `learner_node(agent)` = the node number of the agent's running learner (the latest `VersionInfo.node` of the current generation, or the assignment T2.5 / T4.1 keeps); `collecting = [a for a in agents_with_a_running_learner if collect_allowed(n, learner_node(a), relay=cfg.distributed.relay, path=self._paths.get((n, learner_node(a))))]` is what `OwnerRotation.assign(..., collecting=collecting, ...)` gets for this node's cells; every granted lineup goes through `strip_collect(lineup, collecting)` before `lineup_to_proto`. A node for which `collecting` is empty gets non-collecting lineups (it still plays: anchors, evaluation-free training opponents) unless no agent of the run can collect anywhere (spec 4.2's rule, unchanged).
4. With `distributed.relay: false`, the first time a node reports `down` for a learner node, one WARNING names both nodes ("node <a> cannot reach the data port of node <b> and the relay is off: <a> plays <agents> without collecting"); spec 4.8.

Tests of this step are in Step 8 (`test_relay_off_keeps_an_unreachable_node_from_collecting`, part b) and `test_a_worker_on_another_node_trains_the_learner_directly`.

- [ ] **Step 7: The node routing section (`src/colosseum/node/agent.py`)**

1. On the first `Welcome`: `self._router = ChunkRouter(make_frame=lambda agent, gen, frame, target: pb.ChunkFrame(agent_id=agent, target_node=target, generation=gen, frame=frame), make_channel=<the data-channel factory of T3.1 Step 7>, direct_stub=lambda ch: _with_md(pbg.NodeDataStub(ch).Chunks, data_metadata), relay_call=None (part a; part b: the relay call), connect_timeout_sec=cfg.distributed.connect_timeout_sec, on_path=lambda target, path: self._inbox_put(PathChanged(target, path)))`; `data_metadata` is `[(MD_TOKEN, token.secret)]` with `--insecure`, else empty. The data port of T3.1 gets `delivery=self._router.delivery` (create the router before the data port, or call a `DataServer.set_delivery` — keep one of the two, documented in the code).
2. Routes: whenever the planner's known versions or this node's learners change, `self._router.set_routes({agent: RemoteRoute(info.node, info.generation, tuple(info.addresses), info.relay_only) for agent, info in known if info.node != self._number and agent not in local})`.
3. Worker route connections (T2.2: every worker opens one `route` connection to the node): the node's listener hands each accepted `ChildHello(channel="route")` connection to `self._router.add_worker(conn, worker)` instead of reading it on the main loop (T2.4 may read and drop them today: remove that).
4. Local learners: on every learner start / restart / stop, `self._router.delivery.set_learner(agent, conn)` with the node's chunk connection to that learner incarnation (`ChildHello(role="node", channel="chunks")`, T2.4; open it here if T2.4 does not) or None.
5. `LearnerRoute`s to workers: for every trainable agent: a local learner → its address (as T2.4); a remote learner whose `TargetSender.path` is not `"down"` (or not reported yet) → `LearnerRoute(agent, None, via_node=True, generation)`; else `LearnerRoute(agent, None, via_node=False, generation)` (the worker drops and counts `no_learner` if it still gets a collecting seat). `PathChanged` events recompute them and send the changed ones to every worker.
6. `PathChanged(target, "relay")`: one WARNING per target per node process ("data port of node <t> unreachable at <addresses> within <connect_timeout_sec> s; chunks go through the hub") and one `Event(kind="warning", message=...)` to the hub (logged there with the node name, spec 4.8 "WARNING один раз на пару узлов"). `"down"`: the same with "the relay is off: this node stops collecting for <agents>".
7. `Heartbeat.paths = [PathState(node=t, path=p) for t, p in self._router.paths().items()]`; `Heartbeat.counters` gains `self._router.counters()` (all names of the router and `LocalDelivery`, cumulative for the node process).
8. Stop: `self._router.stop()` before the children are terminated (its threads exit within `_POLL_SEC`; frames still queued are counted `disconnect`).

- [ ] **Step 8: Direct-path cluster test (part a)**

Create `tests/integration/test_sp5_chunks_cluster.py` (`cluster_kit.cluster`, TLS on; nodes `a` with `cpu_learners=1, workers=0` and `b` with `workers=1, no_learners=True`, `learner_first="main"`: 2 loaded processes):
- `test_a_worker_on_another_node_trains_the_learner_directly`: `cluster_config({"training.total_timesteps": 10**9})`; wait until `hub_version(hub, "main") >= 4` (A's learner trains only on B's chunks: A has no worker). Assert: `b.agent.counters()["chunks_sent.direct"] > 0` and `"chunks_sent.relay" not in b.agent.counters()`; no `dropped_chunks.*` counter of `a` or `b` is above 0; `b.agent.fetch_history()["main"]` has at least 2 versions; `b.agent.paths() == {a_number: "direct"}` (`NodeAgent.paths()`: a copy of `router.paths()` published each pass, contract note 10); node b's journaled results exist and some `main` seat has `version_end >= 1`.

Run (a) tests, full fast suite + ruff, commit `feat: chunk routing between nodes over data ports (direct path, backpressure, drop counters)`, push.

#### Part (b): the hub relay

- [ ] **Step 9: Write the failing relay tests**

Create `tests/unit/test_sp5_relay_hub.py`:
```python
"""RelayHub and LatestRelay (SP5 T3.2, spec 4.8): pair buffers with end-to-end credit flow, drops only for an offline
target (disconnect) or a receiver that stopped draining (overflow), round-robin over sources, weight pulls first
and without credit; relayed latest weights uploaded once per version."""
from __future__ import annotations

import threading
import time

from colosseum.hub.relay import LatestRelay, RelayHub
from colosseum.hub.state import VersionInfo


def _hub(**kw):
    hub = RelayHub(**kw)
    hub.set_online({"src": 1, "src2": 3, "dst": 2})
    return hub


def test_a_frame_waits_for_a_credit_and_is_counted_when_sent():
    hub = _hub()
    assert hub.put(1, 2, "f1", 100) is None
    inbox = hub.attach(2)
    assert hub.next_item(inbox, timeout=0.05) is None             # no credit yet
    hub.add_credits(inbox, 1)
    assert hub.next_item(inbox, timeout=0.05) == ("chunk", "f1")
    assert hub.stats() == {"relay_frames": 1, "relay_bytes": 100}


def test_a_full_pair_buffer_blocks_the_sender_until_the_receiver_grants_credit():
    hub = _hub(pair_frames=2, put_timeout_sec=5.0)
    inbox = hub.attach(2)
    assert hub.put(1, 2, "a", 1) is None and hub.put(1, 2, "b", 1) is None
    done = threading.Event()
    threading.Thread(target=lambda: (hub.put(1, 2, "c", 1), done.set()), daemon=True).start()
    assert not done.wait(0.3)                                       # backpressure: the third put waits
    hub.add_credits(inbox, 1)
    assert hub.next_item(inbox, timeout=1.0) == ("chunk", "a")
    assert done.wait(2.0)
    assert hub.stats().get("dropped_chunks.overflow", 0) == 0


def test_a_single_large_frame_is_admitted_into_an_empty_buffer():
    hub = _hub(pair_bytes=10)
    assert hub.put(1, 2, "big", 1000) is None
    assert hub.put(1, 2, "more", 1, should_stop=lambda: True) == "disconnect"   # full: the stream is ending


def test_overflow_when_the_receiver_stops_draining():
    t = [0.0]
    hub = _hub(pair_frames=1, put_timeout_sec=1.0, clock=lambda: t[0])
    assert hub.put(1, 2, "a", 1) is None

    def advance():
        time.sleep(0.05)
        t[0] = 2.0
    threading.Thread(target=advance, daemon=True).start()
    assert hub.put(1, 2, "b", 1) == "overflow"
    assert hub.stats()["dropped_chunks.overflow"] == 1


def test_frames_for_an_offline_node_are_dropped_and_counted_disconnect():
    hub = _hub()
    assert hub.put(1, 9, "x", 1) == "disconnect"
    assert hub.put(1, 2, "a", 1) is None and hub.put(1, 2, "b", 1) is None
    inbox = hub.attach(2)
    hub.set_online({"src": 1})                                     # node 2 lost: its buffers go
    assert inbox.detached
    assert hub.stats()["dropped_chunks.disconnect"] == 3
    assert hub.next_item(inbox, timeout=0.05) is None


def test_sources_are_served_round_robin():
    hub = _hub()
    for i in range(3):
        hub.put(1, 2, f"s1-{i}", 1)
    hub.put(3, 2, "s3-0", 1)
    inbox = hub.attach(2)
    hub.add_credits(inbox, 4)
    got = [hub.next_item(inbox, timeout=0.1)[1] for _ in range(4)]
    assert got == ["s1-0", "s3-0", "s1-1", "s1-2"]


def test_controls_go_first_and_need_no_credit_and_a_new_inbox_detaches_the_old():
    hub = _hub()
    assert hub.push_control(2, "pull") is False                     # no inbox attached
    old = hub.attach(2)
    hub.put(1, 2, "a", 1)
    assert hub.push_control(2, "pull") is True
    assert hub.next_item(old, timeout=0.05) == ("control", "pull")
    new = hub.attach(2)
    assert old.detached and not new.detached
    hub.detach(new)
    assert hub.push_control(2, "pull") is False


def _vi(version, generation=1, node=2):
    return VersionInfo(agent_id="main", generation=generation, version=version, size=4, digest="d", node=node,
                       relay_only=True)


def test_latest_relay_pulls_once_and_serves_every_waiter_from_one_upload(tmp_path):
    hub = _hub()
    inbox = hub.attach(2)
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=5.0)
    relay.publish_versions({"main": _vi(7)})
    results = []

    def download():
        got = relay.open_newer("main", -1)
        if got is not None:
            meta, f = got
            with f:
                results.append((meta.version, f.read()))
    threads = [threading.Thread(target=download) for _ in range(3)]
    for th in threads:
        th.start()
    kind, pull = hub.next_item(inbox, timeout=2.0)
    assert kind == "control" and pull[0] == "main" and pull[1] == -1
    assert hub.next_item(inbox, timeout=0.2) is None                # one pull for three downloads
    spool = tmp_path / "spool"
    spool.write_bytes(b"w7w7")
    assert relay.complete("main", 7, 1, spool) == "accepted"
    for th in threads:
        th.join(5)
    assert results == [(7, b"w7w7")] * 3
    assert relay.open_newer("main", 7, timeout=0.1) is None         # nothing newer is known
    assert [p.name for p in (tmp_path / "relay").iterdir()] == ["latest-main-v7.frame"]


def test_latest_relay_keeps_only_the_newest_and_refuses_stale_uploads(tmp_path):
    hub = _hub()
    hub.attach(2)
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=0.1)
    relay.publish_versions({"main": _vi(8, generation=2)})
    for name, version, generation, expected in (("a", 7, 1, "stale_generation"), ("b", 8, 2, "accepted"),
                                                 ("c", 8, 2, "not_newer")):
        p = tmp_path / name
        p.write_bytes(b"x")
        assert relay.complete("main", version, generation, p) == expected
        assert not p.exists()
    relay.publish_versions({"main": _vi(9, generation=2)})
    p = tmp_path / "d"
    p.write_bytes(b"y")
    assert relay.complete("main", 9, 2, p) == "accepted"
    assert [q.name for q in (tmp_path / "relay").iterdir()] == ["latest-main-v9.frame"]


def test_latest_relay_gives_up_without_a_learner_inbox(tmp_path):
    hub = _hub()
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=5.0)
    relay.publish_versions({"main": _vi(3)})
    assert relay.open_newer("main", -1) is None                      # node 2 has no RelayInbox: no wait
    assert relay.open_newer("other", -1) is None
```
Run → `ModuleNotFoundError: No module named 'colosseum.hub.relay'`.

- [ ] **Step 10: Write `src/colosseum/hub/relay.py`**

```python
"""Chunk and weight relay through the hub (SP5 spec 4.8; T3.2). Thread-safe state shared by the hub's gRPC
servicer threads; the hub main loop only publishes which nodes are online (``set_online``) and the version
registry's latest entries (``LatestRelay.publish_versions``), and reads ``stats()`` on its metrics tick.

Chunks. A sender node streams the frames for one target node with ``Hub.RelaySend``; the target node keeps one
``Hub.RelayInbox`` stream open and grants credits (one per frame it handed to a local learner). Between the two the
hub keeps one bounded buffer per (source, target) pair (``RELAY_PAIR_FRAMES`` frames and ``RELAY_PAIR_BYTES``
bytes; a single frame larger than the byte bound is still admitted into an empty buffer). Frames are never decoded:
the hub forwards the ``ChunkFrame`` message as it came. Flow control is end to end:
- the receiver grants no credit while its learner does not read -> the hub sends nothing -> the pair buffer fills
  -> ``put`` waits -> the RelaySend servicer stops reading -> gRPC flow control stops the sender's stream -> the
  sender's queue fills -> the worker's ``send_bytes`` blocks (the worker loop stalls, as on one machine);
- a frame is dropped only when its target is not online (``disconnect``) or when it found no room for
  ``RELAY_PUT_TIMEOUT_SEC`` (the receiver stopped draining: ``overflow``); both counted.
Round-robin over sources keeps one busy sender from starving another.

Weights. ``LatestRelay`` serves ``Hub.Download(kind="latest")`` for learners whose data port other machines cannot
reach: the first request for a version the hub does not hold sends one ``LatestRequest`` (a "pull") down the
learner node's RelayInbox; the learner node answers with ``Hub.Upload(kind="relay_latest")``; every waiting and
later download of that version is served from the one uploaded file (one upload per version, whatever the number of
fetching nodes). The newest file per agent is kept (``runs/<name>/blobs/relay/``), the previous one deleted after
the replace (open readers keep their file: POSIX).
"""

from __future__ import annotations

import collections
import itertools
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

RELAY_PAIR_FRAMES = 64
RELAY_PAIR_BYTES = 64 << 20
RELAY_PUT_TIMEOUT_SEC = 10.0
RELAY_PULL_WAIT_SEC = 10.0
_WAIT_SLICE_SEC = 0.25


@dataclass
class _Pair:
    items: collections.deque = field(default_factory=collections.deque)   # (item, nbytes)
    nbytes: int = 0


@dataclass
class _Dest:
    """One target node: its pair buffers, its attached inbox, pending control items."""

    cond: threading.Condition
    pairs: dict[int, _Pair] = field(default_factory=dict)
    order: collections.deque = field(default_factory=collections.deque)   # round-robin over source numbers
    inbox: InboxHandle | None = None
    controls: collections.deque = field(default_factory=collections.deque)


@dataclass(eq=False)
class InboxHandle:
    """One RelayInbox stream of a target node (``RelayHub.attach``)."""

    dest: int
    credits: int = 0
    detached: bool = False


class RelayHub:
    """Pair buffers between RelaySend and RelayInbox streams (module docstring)."""

    def __init__(self, *, enabled: bool = True, pair_frames: int = RELAY_PAIR_FRAMES,
                 pair_bytes: int = RELAY_PAIR_BYTES, put_timeout_sec: float = RELAY_PUT_TIMEOUT_SEC,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = bool(enabled)
        self._pair_frames = int(pair_frames)
        self._pair_bytes = int(pair_bytes)
        self._put_timeout = float(put_timeout_sec)
        self._clock = clock
        self._lock = threading.Lock()                  # guards _dests, _online, _numbers and the counters
        self._dests: dict[int, _Dest] = {}
        self._online: set[int] = set()
        self._numbers: dict[str, int] = {}
        self._counters = collections.Counter()

    # --- main loop -----------------------------------------------------------------------------------------------

    def set_online(self, numbers: Mapping[str, int]) -> None:
        """``node_id -> number`` of the nodes with a live session. Buffers towards nodes that left are dropped
        (``disconnect``) and their inboxes detached."""
        with self._lock:
            self._numbers = dict(numbers)
            self._online = set(numbers.values())
            gone = [d for d in self._dests if d not in self._online]
        for number in gone:
            dest = self._dest(number)
            with dest.cond:
                dropped = sum(len(p.items) for p in dest.pairs.values())
                dest.pairs.clear()
                dest.order.clear()
                dest.controls.clear()
                if dest.inbox is not None:
                    dest.inbox.detached = True
                    dest.inbox = None
                dest.cond.notify_all()
            self._count("dropped_chunks.disconnect", dropped)

    def stats(self) -> dict[str, int]:
        """Cumulative counters of this hub process: ``relay_bytes``, ``relay_frames``,
        ``dropped_chunks.overflow``, ``dropped_chunks.disconnect`` (fed into the hub's CounterBook)."""
        with self._lock:
            return dict(self._counters)

    def number_of(self, node_id: str) -> int | None:
        with self._lock:
            return self._numbers.get(node_id)

    # --- sender side (RelaySend servicer threads) -------------------------------------------------------------------

    def put(self, src: int, dst: int, item: Any, nbytes: int, *,
            should_stop: Callable[[], bool] = lambda: False) -> str | None:
        """Queue one frame from node ``src`` for node ``dst``. Blocks while the pair buffer is full (backpressure).
        Returns None when queued, else the drop reason (counted): ``disconnect`` (``dst`` not online, or the
        stream ended while waiting) or ``overflow`` (no room for ``RELAY_PUT_TIMEOUT_SEC``)."""
        with self._lock:
            online = dst in self._online
        if not online:
            self._count("dropped_chunks.disconnect")
            return "disconnect"
        dest = self._dest(dst)
        deadline = self._clock() + self._put_timeout
        with dest.cond:
            while True:
                with self._lock:
                    online = dst in self._online
                if not online or should_stop():
                    reason = "disconnect"
                    break
                pair = dest.pairs.get(src)
                if pair is None:
                    pair = dest.pairs[src] = _Pair()
                    dest.order.append(src)
                if not pair.items or (len(pair.items) < self._pair_frames
                                      and pair.nbytes + nbytes <= self._pair_bytes):
                    pair.items.append((item, int(nbytes)))
                    pair.nbytes += int(nbytes)
                    dest.cond.notify_all()
                    return None
                remaining = deadline - self._clock()
                if remaining <= 0:
                    reason = "overflow"
                    break
                dest.cond.wait(min(remaining, _WAIT_SLICE_SEC))
        self._count(f"dropped_chunks.{reason}")
        return reason

    # --- receiver side (RelayInbox servicer threads) ----------------------------------------------------------------

    def attach(self, dst: int) -> InboxHandle:
        """A RelayInbox stream of node ``dst`` opened; an older stream of the same node is detached (it ends)."""
        dest = self._dest(dst)
        handle = InboxHandle(dst)
        with dest.cond:
            if dest.inbox is not None:
                dest.inbox.detached = True
            dest.inbox = handle
            dest.cond.notify_all()
        return handle

    def detach(self, handle: InboxHandle) -> None:
        dest = self._dest(handle.dest)
        with dest.cond:
            handle.detached = True
            if dest.inbox is handle:
                dest.inbox = None
            dest.cond.notify_all()

    def add_credits(self, handle: InboxHandle, frames: int) -> None:
        if frames <= 0:
            return
        dest = self._dest(handle.dest)
        with dest.cond:
            handle.credits += int(frames)
            dest.cond.notify_all()

    def push_control(self, dst: int, item: Any) -> bool:
        """A control item (a weights pull) for node ``dst``; sent before chunks and without credit. False when the
        node has no inbox attached."""
        dest = self._dest(dst)
        with dest.cond:
            if dest.inbox is None:
                return False
            dest.controls.append(item)
            dest.cond.notify_all()
            return True

    def next_item(self, handle: InboxHandle, timeout: float) -> tuple[str, Any] | None:
        """The next item for ``handle``'s node: ``("control", item)`` or ``("chunk", item)`` (one credit spent);
        None after ``timeout`` or when the handle was detached (check ``handle.detached``)."""
        dest = self._dest(handle.dest)
        deadline = self._clock() + timeout
        with dest.cond:
            while not handle.detached:
                if dest.controls:
                    return "control", dest.controls.popleft()
                if handle.credits > 0:
                    picked = self._pop_round_robin(dest)
                    if picked is not None:
                        item, nbytes = picked
                        handle.credits -= 1
                        dest.cond.notify_all()        # room for a waiting put
                        self._count("relay_frames")
                        self._count("relay_bytes", nbytes)
                        return "chunk", item
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return None
                dest.cond.wait(min(remaining, _WAIT_SLICE_SEC))
        return None

    # --- internals -----------------------------------------------------------------------------------------------

    def _dest(self, number: int) -> _Dest:
        with self._lock:
            dest = self._dests.get(number)
            if dest is None:
                dest = self._dests[number] = _Dest(threading.Condition(threading.Lock()))
            return dest

    @staticmethod
    def _pop_round_robin(dest: _Dest) -> tuple[Any, int] | None:
        for _ in range(len(dest.order)):
            src = dest.order[0]
            dest.order.rotate(-1)
            pair = dest.pairs[src]
            if pair.items:
                item, nbytes = pair.items.popleft()
                pair.nbytes -= nbytes
                return item, nbytes
        return None

    def _count(self, name: str, n: int = 1) -> None:
        if n:
            with self._lock:
                self._counters[name] += n


@dataclass(frozen=True)
class RelayedVersion:
    agent_id: str
    version: int
    generation: int
    path: Path
    teacher_active: bool = False


class LatestRelay:
    """Latest weights of relay-only learners, uploaded once per version on demand (module docstring)."""

    def __init__(self, root: Path, relay: RelayHub, *, wait_sec: float = RELAY_PULL_WAIT_SEC,
                 make_pull: Callable[[str, int, str], Any] = lambda agent, since, rid: (agent, since, rid),
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._relay = relay
        self._wait = float(wait_sec)
        self._make_pull = make_pull
        self._clock = clock
        self._cond = threading.Condition()
        self._versions: dict[str, Any] = {}               # agent -> VersionInfo (registry latest)
        self._cached: dict[str, RelayedVersion] = {}
        self._pending: dict[str, tuple[int, float]] = {}  # agent -> (version asked, when)
        self._ids = itertools.count(1)

    def publish_versions(self, versions: Mapping[str, Any]) -> None:
        """Main loop: the registry's latest ``VersionInfo`` per agent (after every accepted announcement)."""
        with self._cond:
            self._versions = dict(versions)
            self._cond.notify_all()

    def latest(self, agent_id: str) -> Any | None:
        with self._cond:
            return self._versions.get(agent_id)

    def open_newer(self, agent_id: str, since_version: int, *,
                   timeout: float | None = None) -> tuple[RelayedVersion, BinaryIO] | None:
        """An open file of the newest relayed version above ``since_version``, pulling it from the learner node
        when the hub holds none (one pull in flight per agent; a pull unanswered for ``wait_sec`` is resent).
        None after ``timeout`` (default ``wait_sec``)."""
        timeout = self._wait if timeout is None else float(timeout)
        deadline = self._clock() + timeout
        with self._cond:
            while True:
                cached = self._cached.get(agent_id)
                if cached is not None and cached.version > since_version:
                    try:
                        return cached, open(cached.path, "rb")
                    except FileNotFoundError:     # replaced between the lookup and the open: look again
                        continue
                info = self._versions.get(agent_id)
                if info is None or info.version <= since_version:
                    return None
                now = self._clock()
                pending = self._pending.get(agent_id)
                if pending is None or pending[0] < info.version or now - pending[1] >= self._wait:
                    pull = self._make_pull(agent_id, cached.version if cached else -1, f"pull-{next(self._ids)}")
                    if self._relay.push_control(info.node, pull):
                        self._pending[agent_id] = (info.version, now)
                    else:
                        return None                   # the learner node has no relay inbox: nothing to wait for
                remaining = deadline - now
                if remaining <= 0:
                    return None
                self._cond.wait(min(remaining, _WAIT_SLICE_SEC))

    def complete(self, agent_id: str, version: int, generation: int, spooled: Path, *,
                 teacher_active: bool = False) -> str:
        """An upload of ``agent_id`` v``version`` arrived (digest and frame header already checked;
        ``teacher_active`` read from the header). ``accepted`` (it becomes the served file), ``not_newer`` or
        ``stale_generation`` (the file is deleted)."""
        with self._cond:
            info = self._versions.get(agent_id)
            cached = self._cached.get(agent_id)
            if info is not None and generation != info.generation:
                reason = "stale_generation"
            elif cached is not None and version <= cached.version:
                reason = "not_newer"
            else:
                final = self._root / f"latest-{agent_id}-v{int(version)}.frame"
                os.replace(spooled, final)
                self._cached[agent_id] = RelayedVersion(agent_id, int(version), int(generation), final,
                                                        bool(teacher_active))
                if cached is not None and cached.path != final:
                    cached.path.unlink(missing_ok=True)
                pending = self._pending.get(agent_id)
                if pending is not None and pending[0] <= version:
                    self._pending.pop(agent_id, None)
                self._cond.notify_all()
                return "accepted"
        spooled.unlink(missing_ok=True)
        return reason
```
Lock order (no deadlock): a destination's condition first, then the hub-wide lock (`_count`, `online`); `set_online` takes the hub-wide lock alone and releases it before it touches a destination. Run the relay tests → 10 passed.

- [ ] **Step 11: The relay servicers and relayed weights (`src/colosseum/hub/server.py`)**

`build_server(..., relay: RelayHub | None = None, latest: LatestRelay | None = None)`; the hub passes `RelayHub(enabled=cfg.distributed.relay)` and `LatestRelay(run_dir.root / "blobs" / "relay", relay, wait_sec=RELAY_PULL_WAIT_SEC, make_pull=lambda agent, since, rid: pb.LatestRequest(agent_id=agent, since_version=since, request_id=rid))`. The servicer's caller identity is the node id the `AuthInterceptor` verified (`x-colosseum-node` metadata); its number is `relay.number_of(node_id)` (None → `PERMISSION_DENIED`, "unknown or offline node"). The server's thread pool must hold one thread per open session, relay inbox and relay send stream: at least 4 × the design's 20 nodes + 32 (if T2.3 sized it lower, raise it; one constant).

```python
    def RelaySend(self, request_iterator, context):  # noqa: N802
        src = self._caller_number(context)
        if not self._relay.enabled:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "distributed.relay is false on this hub")
        received = dropped = 0
        for frame in request_iterator:
            received += 1
            if self._relay.put(src, frame.target_node, frame, len(frame.frame),
                               should_stop=lambda: not context.is_active()) is not None:
                dropped += 1
        return pb.ChunkSummary(received=received, dropped=dropped)

    def RelayInbox(self, request_iterator, context):  # noqa: N802
        dst = self._caller_number(context)
        if not self._relay.enabled:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "distributed.relay is false on this hub")
        first = next(request_iterator, None)
        if first is None:
            return
        handle = self._relay.attach(dst)
        self._relay.add_credits(handle, first.frames)

        def read_credits() -> None:
            try:
                for credit in request_iterator:
                    self._relay.add_credits(handle, credit.frames)
            except grpc.RpcError:
                pass
            finally:
                self._relay.detach(handle)
        threading.Thread(target=read_credits, name=f"relay-credits-n{dst}", daemon=True).start()
        try:
            while context.is_active() and not handle.detached:
                got = self._relay.next_item(handle, timeout=0.25)
                if got is None:
                    continue
                kind, item = got
                yield pb.RelayItem(chunk=item) if kind == "chunk" else pb.RelayItem(pull=item)
        finally:
            self._relay.detach(handle)
```
`Download` gains `kind == "latest"`: `got = self._latest.open_newer(request.agent_id, request.since_version)`; None → if `latest(agent)` exists and its version ≤ `since_version`: one `BlobPart(not_modified=True, last=True)`; else `context.abort(UNAVAILABLE, "weights of <agent> are not on the hub yet; retried at the next sync")`; else stream the open file through an `mmap` with `send_parts(..., meta_json=json.dumps({"agent_id", "version", "generation", "size", "digest", "teacher_active"}))` (the same metadata as `GetLatest`) and count the bytes into the relay's `relay_bytes`.
`Upload` gains `kind == "relay_latest"` handled in the servicer thread (no `UploadIn` event: no hub state is touched): the caller must be the agent's learner node (`relay.number_of(caller) == latest(agent).node`, else `accepted=False, reason="not_learner_node"`); spool to `blobs/relay/.incoming-<uuid>` with the size cap `payload_byte_cap(max_message_mb)` and `total_size`; compute blake2b-16 while spooling and compare with `header.digest`; `peek_meta` the spooled file's header (agent and `policy_version` must match the header; `teacher_active` read from it); then `reason = self._latest.complete(agent, version, generation, path, teacher_active=header_meta["teacher_active"])` → `UploadReply(accepted=reason == "accepted", reason=reason)`. Any failure deletes the spool file.

Create `tests/integration/test_sp5_relay_servicers.py` (the real hub server of `build_server` with `hub_kit` fake nodes (T2.3) as in-process clients; the test publishes `relay.set_online` and `latest.publish_versions` itself, standing in for the main loop):
- `test_relay_streams_frames_to_the_target_with_credits`: fake node 1 streams 50 `ChunkFrame`s for target 2 with `RelaySend`; fake node 2 opens `RelayInbox` with 16 credits and grants one per frame read → it reads all 50 frames byte-identical, in order per source; `relay.stats()["relay_frames"] == 50`, `relay_bytes` = their sizes; RelaySend's summary `received == 50, dropped == 0`.
- `test_frames_for_an_offline_target_are_dropped_and_counted`: target 9 not in `set_online` → summary `dropped == n`, `stats()["dropped_chunks.disconnect"] == n`.
- `test_relayed_latest_weights_are_uploaded_once_for_many_downloads`: `publish_versions({"main": VersionInfo(node=2, version=5, relay_only=True, ...)})`; fake node 2's `RelayInbox` answers every pull item by uploading `relay_latest` (a weights frame of v5 with its digest); three fake nodes call `Download(kind="latest", since_version=-1)` at once → all three receive the same frame (digest checked by `read_latest_reply`), node 2 saw exactly one pull, `blobs/relay/` holds one file; an upload by node 1 for `main` is refused `not_learner_node`; an upload with a wrong digest is refused and leaves no file.
- `test_relay_off_refuses_relay_streams`: `RelayHub(enabled=False)` → `RelaySend` and `RelayInbox` fail `FAILED_PRECONDITION`.

- [ ] **Step 12: The node's relay pieces (`src/colosseum/node/agent.py`, `node/dataport.py`)**

1. `node/dataport.py`:
```python
def fetch_latest_relay(channel: grpc.Channel, agent_id: str, since_version: int, *, max_message_mb: int,
                       metadata: Sequence[tuple[str, str]], timeout: float = 60.0) -> FetchedLatest | None:
    """Latest weights of a learner whose data port this node cannot reach, through the hub
    (``Hub.Download(kind="latest")``; spec 4.8). None: not modified. Raises ``FetchError``."""
    stub = pbg.HubStub(channel)
    try:
        parts = stub.Download(pb.BlobRequest(kind="latest", agent_id=agent_id, since_version=int(since_version)),
                              metadata=tuple(metadata), timeout=timeout)
    except grpc.RpcError as e:
        raise FetchError("rpc", f"{e.code()}: {e.details()}") from e
    return read_latest_reply(parts, agent_id, max_message_mb=max_message_mb, source="relay")
```
2. `_fetch_job` (T3.1): when the source is `relay_only`, or `probe_direct` finds no address within `connect_timeout_sec` (remembered per learner node for `DIRECT_RETRY_SEC`, like the router), and `cfg.distributed.relay` is on and the session is up → `fetch_latest_relay(self._session.channel, ...)` with the node's R1 metadata; an `UNAVAILABLE` "not on the hub yet" is a failed job (retried at the next interval; not a WARNING, counted `weight_fetch_failed`). Relay off and no direct path → `LatestFetchFailed(agent, "unreachable")` (T3.1's path; WARNING once).
3. `self._router`'s `relay_call` (part a: None) becomes `lambda: _with_md(pbg.HubStub(self._session.channel).RelaySend, self._auth_md)` when `cfg.distributed.relay` is on.
4. `RelayReceiver`: started when this node runs at least one learner and the relay is on (stopped when it runs none): `open_inbox=lambda it: pbg.HubStub(self._session.channel).RelayInbox(it, metadata=self._auth_md)`, `make_credit=lambda n, node_id: pb.RelayCredit(frames=n, node_id=node_id)`, `delivery=self._router.delivery`, `chunk_of=lambda item: (item.chunk.agent_id, item.chunk.frame) if item.WhichOneof("i") == "chunk" else None`, `on_pull=lambda item: self._inbox_put(RelayPullIn(item.pull))`, `counters=self._router.counters_obj`.
5. `_on_relay_pull(ev)`: the main loop submits `_upload_latest(ev.request)` to the transfer pool (at most one in flight per agent; a pull while one runs is ignored: the running upload sends the newest anyway): `opened = self._cache_reader.open_latest(agent)` (None → nothing to send); stream `UploadPart(header=UploadHeader(kind="relay_latest", agent_id, generation, version, total_size, digest, request_id))` followed by `UploadPart(data=...)` parts of `PART_BYTES` read from an `mmap` of the open file; a refusal is logged at INFO (`not_newer` is normal), counted `relay_uploads_refused`; `relay_uploads` counts accepted ones.

- [ ] **Step 13: Hub wiring of the relay (`src/colosseum/hub/hub.py`)**

1. `_on_session_opened` / `_on_session_closed` / node loss: `self._relay.set_online({entry.node_id: entry.number for entry in registry.online()})`.
2. After every accepted version (`_on_version`, T3.1 Step 8): `self._latest.publish_versions({agent: registry.latest(agent) for agent in agents with a version})`.
3. Metrics tick: `self._book.report(f"hub/{self._incarnation}", self._relay.stats())`, with `self._incarnation = uuid.uuid4().hex` per hub process (so a resumed hub adds to the totals instead of replacing them); `hub_state.json` `counters`: `relay_bytes = book.get("relay_bytes")` and `dropped_chunks = book.grouped(groups={"dropped_chunks": DROP_REASONS})["dropped_chunks"]` (`DROP_REASONS = ("disconnect", "no_learner", "stale", "stale_generation", "invalid", "overflow")`, defined in `hub.state`); the learner-side reasons (`invalid`, `stale`, `stale_generation`, learner-side `disconnect`) come from T3.3's `MetricsHub` totals and are added by T7.1 into the `system` record (and by T4.3 into the resume state; contract note 17).

- [ ] **Step 14: Relay cluster tests**

Extend `tests/integration/test_sp5_chunks_cluster.py` (same two nodes). Node a's data port is announced behind a `TcpFaultProxy()` created without a target: `start_node(..., data_addresses=[proxy.address])`, then, once node a has bound its port, `proxy.set_target(("127.0.0.1", a.agent.data_port()))` (`NodeAgent.data_port() -> int | None`, the bound port, contract note 10):
- `test_relay_carries_chunks_and_weights_when_the_data_port_is_unreachable`: `proxy.stall()` before node b starts (`connect_timeout_sec` 1.0 in `cluster_config`); wait until `hub_version(hub, "main") >= 4`. Assert: `b`'s counters have `chunks_sent.relay > 0` and no `chunks_sent.direct`; the relay carried weights too (`b`'s fetch history has ≥ 2 versions and the hub's `blobs/relay/` holds a `latest-main-v*.frame`); the hub book's `relay_bytes > 0` after a metrics tick; node b logged exactly one WARNING naming node a's number and "through the hub"; all `dropped_chunks.*` of both nodes and the relay are 0.
- `test_relay_off_keeps_an_unreachable_node_from_collecting`: `cluster_config({"distributed.relay": False})`, `proxy.stall()`; node b plays (journaled results of node b grow) but A's learner never trains (`hub_version(hub, "main")` stays 0 for 15 s of node-b matches); node b's counters show at most the frames made before its path report as `dropped_chunks.disconnect` (≤ `SENDER_QUEUE_FRAMES` + one stream) and then no growth; the hub logged the "relay is off" WARNING once.

Create `tests/integration/test_sp5_chunks_cluster_slow.py` (`@pytest.mark.slow`; same two nodes, 2 loaded):
- `test_a_cut_is_survived_and_every_chunk_is_counted`: monkeypatch `colosseum.node.router.DIRECT_RETRY_SEC` to 3.0 (the nodes run in the test process); direct path first (proxy normal, `chunks_sent.direct > 0`), then `proxy.cut(5.0)`: the sender switches to the relay within `connect_timeout_sec` + 1 s (`b.agent.counters()["chunks_sent.relay"]` grows while the cut lasts) and A's learner keeps training (`hub_version` grows during the cut); after the cut, within `DIRECT_RETRY_SEC` + `STREAM_ROTATE_SEC` + 2 s the path is direct again (`paths()[a] == "direct"` and `chunks_sent.direct` grows). Accounting: `b`'s `dropped_chunks.disconnect` ≤ `STREAM_ROTATE_FRAMES + SENDER_QUEUE_FRAMES` (one broken stream, counted as an upper bound) and every chunk B's worker reported sent (`WorkerProgress.counters["chunks_sent"]`, forwarded in the node's worker stats) ≤ the frames A's intake received + every counted drop on B, the relay and A (no uncounted loss).
- `test_a_relay_only_learner_survives_a_cut_of_the_hub_relay_stream`: `proxy.stall()` (relay only); kill node b's relay streams by cutting nothing on the hub (the hub port is not proxied): instead `relay.set_online` drops b for one pass (call `hub.hub._relay.set_online({...without b...})` then restore) → frames in the pair buffers to B are counted `disconnect`, B's `RelayReceiver` reattaches (`connected` set again) and training goes on.

- [ ] **Step 15: Run, full fast suite, commit and push part (b)**

Run the relay tests and the cluster tests (`-m slow` ones alone, on a quiet machine), the full fast suite and ruff. Commit `feat: hub relay for chunks and weights (pair buffers with credits, one upload per version, relay: false)`; push.

---

### Task T3.3: `learner.max_policy_lag`, policy-lag percentiles, `dropped_stale_chunks`

Spec 4.7 ("`learner.max_policy_lag` (в версиях; `null` — выключено, по умолчанию): лёрнер сбрасывает чанки с `policy_version` старше `текущая − max_policy_lag` (`dropped_chunks.stale`). Лаг (`текущая − policy_version`) p50 / p95 / max пишется в `train` всегда"), spec 8 (`train`: `policy_lag_p50` / `p95` / `max`, `dropped_stale_chunks`). The learner's intake (T2.2: `ConnectionIntake` on a node, `QueueIntake` in the queue-path tests and the old roles) learns the learner's current version before each batch (`note_policy_version`) and, with `max_policy_lag` set, drops every chunk whose `policy_version < current − max_policy_lag` before decoding it (the frame header carries the version), counted `stale`. The learner reports the lag distribution of every trained batch and the cumulative stale count. `MetricsHub` keeps the learner-side drop reasons of every learner incarnation (from `IntakeStats`) as monotonic run totals. A config that would starve the learner (`max_policy_lag < learner.weight_push_interval − 1`) is a `ConfigError`.

**Files:**
- Modify: `src/colosseum/learner/learner.py` (`lag_metrics`, the loop's `note_policy_version` call and metrics)
- Modify: `src/colosseum/net/local.py` (`ConnectionIntake` / `QueueIntake`: `max_policy_lag`, `note_policy_version`; the `ChunkIntake` protocol)
- Modify: `src/colosseum/node/learner_main.py` (passes `config.max_policy_lag` to `ConnectionIntake`) — one keyword in T2.4's construction of the intake; no other change there
- Modify: `src/colosseum/metrics/hub.py` (learner-side chunk drops per incarnation, `chunk_drops()`, `intake_book`)
- Modify: `src/colosseum/core/config.py` (one `LearnerConfig` validator)
- Test: `tests/unit/test_sp5_policy_lag.py`, `tests/integration/test_sp5_policy_lag_run.py`; extended: `tests/unit/test_learner_v2.py` (unchanged assertions must keep passing), the T1.4 config tests (`tests/unit/test_sp5_config.py` or its part-01 name)

**Interfaces:**
- Consumes: `ChunkIntake`, `ConnectionIntake`, `QueueIntake`, `learner_process(intake=..., outputs=...)`, `collect_batch` (T2.2); `LearnerConfig.max_policy_lag` (T1.4); `peek_meta` (T1.1); `CounterBook` (T3.1); the hub's forwarding of `IntakeStats` to `MetricsHub` (T2.5, assumption A6).
- Produces (contract additions): `ChunkIntake.note_policy_version(version: int) -> None`; `ConnectionIntake(..., max_policy_lag: int | None = None)`, `QueueIntake(..., max_policy_lag: int | None = None)`; `colosseum.learner.learner.lag_metrics(current: int, versions: Sequence[int]) -> dict[str, float]`; `train` keys `policy_lag_p50`, `policy_lag_p95` (and the kept `policy_lag_mean`, `policy_lag_max`), `dropped_stale_chunks`; `MetricsHub.on_intake_stats(agent_id, generation, depth, dropped)` (or T2.5's existing method of that role, extended), `MetricsHub.chunk_drops() -> dict[str, int]`, `MetricsHub.intake_book: CounterBook`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_policy_lag.py`:
```python
"""Policy lag on the learner (SP5 T3.3, spec 4.7, 8): lag percentiles of every trained batch, max_policy_lag drops
too-old chunks before decoding (counted stale), the hub keeps learner-side drops monotonic, a starving
configuration is refused."""
from __future__ import annotations

import queue
import threading
import time

import pytest

from colosseum.core.config import LearnerConfig
from colosseum.core.errors import ConfigError
from colosseum.learner.learner import lag_metrics, learner_process
from colosseum.net.local import QueueIntake, QueueOutputs
from game_helpers import NumpyOnlyQueue, RecordingLearnerAlgorithm, chunk_v2_payload


def test_lag_metrics_are_percentiles_of_the_batch():
    m = lag_metrics(10, [7, 10, 9])                      # lags 3, 0, 1
    assert m == pytest.approx({"policy_lag_mean": 4 / 3, "policy_lag_p50": 1.0, "policy_lag_p95": 2.8,
                               "policy_lag_max": 3.0})


def _run(algo, payloads, until, *, batch_chunks, max_policy_lag=None, timeout=10.0):
    traj, metrics, stop = NumpyOnlyQueue(), NumpyOnlyQueue(), threading.Event()
    for payload in payloads:
        traj.put(payload)
    intake = QueueIntake(traj, agent_id="a", max_policy_lag=max_policy_lag)
    outputs = QueueOutputs(weight_queues=[NumpyOnlyQueue(maxsize=1)], metrics_queue=metrics)
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=lambda: algo, intake=intake, outputs=outputs,
        config=LearnerConfig(batch_chunks=batch_chunks, device="cpu", max_policy_lag=max_policy_lag,
                             weight_push_interval=1),
        stop_event=stop))
    thread.start()
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.02)
    time.sleep(0.2)                                    # let the loop take what is left in the queue
    stop.set()
    thread.join(10)
    assert not thread.is_alive()
    return intake, [metrics.get_nowait() for _ in range(metrics.qsize())]


def test_the_learner_reports_lag_percentiles_and_keeps_the_old_keys():
    algo = RecordingLearnerAlgorithm(start_version=10)
    _, metrics = _run(algo, [chunk_v2_payload(version=v) for v in (7, 10, 9)], lambda: len(algo.batches) >= 1,
                      batch_chunks=3)
    m = metrics[0]
    assert (m["policy_lag_mean"], m["policy_lag_p50"], m["policy_lag_p95"], m["policy_lag_max"]) == \
        pytest.approx((4 / 3, 1.0, 2.8, 3.0))
    assert m["dropped_stale_chunks"] == 0


def test_max_policy_lag_drops_old_chunks_and_counts_them_stale():
    algo = RecordingLearnerAlgorithm(start_version=10)
    intake, metrics = _run(algo, [chunk_v2_payload(version=v) for v in (7, 10, 9, 10, 10, 8)],
                           lambda: len(algo.batches) >= 2, batch_chunks=2, max_policy_lag=1)
    assert algo.behavior_versions[:2] == [[10, 9], [10, 10]]     # 7 is older than 10 - 1; 8 older than 11 - 1
    assert metrics[0]["dropped_stale_chunks"] == 1
    assert metrics[0]["policy_lag_max"] == 1.0 and metrics[1]["policy_lag_max"] == 1.0
    assert intake.dropped["stale"] == 2                          # 8 arrived after the second batch


def test_without_max_policy_lag_nothing_is_dropped():
    algo = RecordingLearnerAlgorithm(start_version=50)
    intake, metrics = _run(algo, [chunk_v2_payload(version=v) for v in (0, 1)], lambda: len(algo.batches) >= 1,
                           batch_chunks=2)
    assert algo.behavior_versions[0] == [0, 1] and intake.dropped.get("stale", 0) == 0
    assert metrics[0]["policy_lag_max"] == 50.0


def test_a_lag_bound_below_the_push_interval_minus_one_is_refused():
    with pytest.raises((ConfigError, ValueError), match="max_policy_lag"):
        LearnerConfig(max_policy_lag=2, weight_push_interval=5)
    assert LearnerConfig(max_policy_lag=4, weight_push_interval=5).max_policy_lag == 4
    assert LearnerConfig(max_policy_lag=0, weight_push_interval=1).max_policy_lag == 0
```
Plus, in the same file:
- `test_connection_intake_drops_stale_frames_before_decoding`: build a `ConnectionIntake` the way T2.2's tests do (a `net.local.listener` with a fresh name and authkey, a control connection pair, a sender connection that sends `ChildHello(channel="chunks")` and then frames); frames: `encode_chunk` of chunks with `policy_version` 3, 9, 10, and a frame whose header says version 2 but whose data region is truncated; `intake.note_policy_version(10)` with `max_policy_lag=1`; `get(timeout=2)` twice returns the chunks of versions 9 and 10 in order; `dropped["stale"] == 2` (versions 3 and 2: the truncated one is dropped by its header before decoding — documented: a stale frame is never decoded, so its body is not checked); `dropped.get("invalid", 0) == 0`; with `note_policy_version(20)` a later version-9 frame is stale too (the bound follows the learner).
- `test_metrics_hub_sums_learner_side_chunk_drops_over_incarnations`: a `MetricsHub` as in T2.5's tests; `on_intake_stats("main", 1, 3, {"stale": 3})`, `("main", 1, 0, {"stale": 5, "invalid": 1})`, `("main", 2, 0, {"stale": 1})` → `chunk_drops() == {"stale": 6, "invalid": 1}` (5 from generation 1 plus 1 from generation 2); `intake_book.state()` round-trips through `CounterBook.from_state`.

Create `tests/integration/test_sp5_policy_lag_run.py`:
- `test_max_policy_lag_bounds_the_lag_of_every_trained_batch`: `cli_runner.run_train(TTT_CONFIG, tmp_path, overrides={"learner.max_policy_lag": 2, "learner.weight_push_interval": 1})` (1 worker: 2 loaded processes) → exit 0; every `train` record has `policy_lag_max <= 2`, the keys `policy_lag_p50`, `policy_lag_p95`, `dropped_stale_chunks`; the last record's `dropped_stale_chunks >= 0` and the run reached the budget (no starvation). Without the override (a second run is not needed: `test_learner_v2.py` and the T0.1 contract runs cover the default), `max_policy_lag` stays off.

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_policy_lag.py tests/integration/test_sp5_policy_lag_run.py -q` → `ImportError: cannot import name 'lag_metrics'` and `TypeError` on `max_policy_lag`.

- [ ] **Step 2: The intakes (`src/colosseum/net/local.py`)**

1. `ChunkIntake` (protocol) gains `def note_policy_version(self, version: int) -> None` ("the learner's current policy version, before each batch: the reference of `max_policy_lag`").
2. `ConnectionIntake.__init__(..., max_policy_lag: int | None = None)`; `self._min_version = None`; `note_policy_version(v)`: `self._min_version = None if self._max_lag is None else int(v) - self._max_lag`.
3. `ConnectionIntake.get(timeout)`: for each frame taken from the deque (T2.2's loop over frames until a valid chunk or the timeout), after T2.2's header checks (`peek_meta`, generation rule) and **before** `decode_chunk`: `if self._min_version is not None and int(meta["policy_version"]) < self._min_version: self._drop("stale"); continue` (`_drop` = T2.2's per-reason counter that `IntakeReport.dropped` reports). The timeout budget is the whole call's (a run of stale frames never extends it).
4. `QueueIntake` gets the same keyword and method; its check runs on the decoded chunk's `policy_version` (queue items are payload dicts or chunks, already in memory).

- [ ] **Step 3: The learner (`src/colosseum/learner/learner.py`)**

1. New module function next to `collect_batch`:
```python
def lag_metrics(current: int, versions: Sequence[int]) -> dict[str, float]:
    """Policy lag (``current - policy_version``) of the chunks of one trained batch (spec 4.7, 8): mean, p50, p95
    (``numpy.percentile``, linear interpolation) and max. Written to every ``train`` record."""
    lags = np.asarray([int(current) - int(v) for v in versions], dtype=np.float64)
    return {"policy_lag_mean": float(lags.mean()), "policy_lag_p50": float(np.percentile(lags, 50)),
            "policy_lag_p95": float(np.percentile(lags, 95)), "policy_lag_max": float(lags.max())}
```
2. `learner_process`: before every `collect_batch(intake, ...)` call `intake.note_policy_version(algorithm.policy_version)` (the version only changes after a train step, so the bound is exact for the whole batch); replace the `lags` / `policy_lag_mean` / `policy_lag_max` lines by `lag = lag_metrics(algorithm.policy_version, [c.policy_version for c in chunks])` computed before the train step and `metrics.update(lag)` after it; add `metrics["dropped_stale_chunks"] = int(intake.dropped.get("stale", 0))` (cumulative for this learner incarnation; the run total is the hub's, Step 5). The module docstring's lag sentence says: "`train` records carry the lag mean / p50 / p95 / max of the batch and `dropped_stale_chunks`; with `learner.max_policy_lag` the intake drops chunks older than `current - max_policy_lag` before decoding them (spec 4.7)."
3. `node/learner_main.py`: the `ConnectionIntake(...)` construction passes `max_policy_lag=config.max_policy_lag` (the agent's merged `LearnerConfig`).
`tests/unit/test_learner_v2.py::test_learner_reports_policy_lag_from_chunk_versions` keeps passing unchanged (mean and max per batch are the same numbers).

- [ ] **Step 4: The starving configuration (`src/colosseum/core/config.py`)**

`LearnerConfig` gets a `model_validator(mode="after")`: when `max_policy_lag is not None and max_policy_lag < weight_push_interval - 1`, raise `ValueError` (pydantic wraps it; `load_config` turns it into the usual `ConfigError`) with: "learner.max_policy_lag N is below learner.weight_push_interval - 1 (P - 1): after a weight push the learner can take at most N + 1 steps on chunks of the pushed version, so it would wait for chunks that never come; raise max_policy_lag to at least P - 1 or lower weight_push_interval". Why: the learner pushes weights every `weight_push_interval` train steps; chunks made with version v stay admissible while the learner's version ≤ v + N; with N + 1 < P it stops before the next push for ever. The per-agent override merge validates each agent's merged `LearnerConfig`, so the check also covers `agents.<id>.learner`. Add `test_max_policy_lag_below_the_push_interval_is_a_config_error` to the T1.4 config tests (a global and a per-agent case, both `ConfigError` naming the key). `scripts/bench_throughput.py::_make_config` needs no change (it leaves `max_policy_lag` unset).

- [ ] **Step 5: Learner-side drops in the hub (`src/colosseum/metrics/hub.py`)**

`MetricsHub.__init__` creates `self.intake_book = CounterBook()`; the method T2.5's `_on_metrics_batch` calls for each `IntakeStats` (named `on_intake_stats(agent_id, generation, depth, dropped)` here; if T2.5 named it otherwise, extend that one) keeps the depth as before and calls `self.intake_book.report(f"learner/{agent_id}/{generation}", {f"dropped_chunks.{k}": int(v) for k, v in dropped.items()})`; `chunk_drops()` returns `{k.split(".", 1)[1]: v for k, v in self.intake_book.totals().items() if k.startswith("dropped_chunks.")}`. T7.1 adds these to the node-side and relay totals of the hub's book for `system.dropped_chunks`; T4.3 saves and restores `intake_book.state()` in `hub_state.json` (contract note 17).

- [ ] **Step 6: Run the new and touched tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_policy_lag.py tests/integration/test_sp5_policy_lag_run.py tests/unit/test_learner_v2.py -q -rw` → pass; also the T1.4 config tests and the T2.2 intake tests.

- [ ] **Step 7: Full fast suite + ruff; slow suite once**

Full fast suite and ruff (green, zero warnings). This task changes the learner loop: run `env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v` once, alone, on a quiet machine, in the background (every SP2–SP4 threshold unchanged: `max_policy_lag` is off by default, the lag keys are additions).

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/learner/learner.py src/colosseum/net/local.py src/colosseum/node/learner_main.py \
        src/colosseum/metrics/hub.py src/colosseum/core/config.py tests/unit/test_sp5_policy_lag.py \
        tests/integration/test_sp5_policy_lag_run.py tests/unit/test_sp5_config.py
git commit -m "feat: learner.max_policy_lag (stale chunks dropped before decoding) and policy-lag percentiles in train"
git push
```

---

## Contract notes

### Additions and deviations versus `00-overview.md`

1. **New module `colosseum.metrics.counters`** (T3.1): `CounterBook(totals=None, sources=None)` with `report(source, values) -> dict[str, int]`, `add(name, delta)`, `get(name)`, `totals()`, `grouped(groups=...)`, `state()`, `from_state(d)`. One rule for every monotonic run total (spec 8); sources are `"<node_id>/<session_id>"` (node heartbeats), `"hub/<hub incarnation uuid>"` (relay stats), `"learner/<agent>/<generation>"` (intake drops in `MetricsHub`).
2. **New module `colosseum.node.fetch`** (T3.1): `FetchPlanner(node=, sync_interval_sec=)` (`on_version`, `set_local_agents`, `set_referenced`, `forget`, `known`, `in_flight`, `due(cached, hub_online=, now=)`, `finished(agent, version, cached=)`, `failed(agent)`, `stats`), `FetchJob`, `FetchStats` (`as_counters()`).
3. **New module `colosseum.node.dataport`** (T3.1; `Chunks` T3.2): `DataServer` (the `NodeData` servicer and its server; `start(port, credentials=, host=) -> int`, `stop`, `set_node`, `counters`), `probe_direct`, `fetch_latest`, `fetch_latest_relay` (T3.2), `read_latest_reply`, `check_fetched`, `FetchError(reason)`, `FetchedLatest`, `data_addresses(port, hub_host)`, `frame_digest`, `DATA_PORT_TRIES = 16`, `DATA_SERVER_THREADS = 64`. **Deviation:** the overview puts `DataServer` into `node/router.py` (T3.2); T3.1 needs it first (`GetLatest`), and the router module keeps only the outbound path.
4. **`colosseum.node.router`** (T3.2) has more names than the overview's two: `ChunkRouter` (`routes`, `delivery`, `set_routes`, `sender_for`, `add_worker`, `paths`, `counters`, `counters_obj`, `stop`), `RouteTable`, `RemoteRoute(node, generation, addresses, relay_only)`, `TargetSender`, `RouteReader`, `LocalDelivery` (`set_learner`, `deliver`, `count`), `RelayReceiver`, `Counters`; constants `SENDER_QUEUE_FRAMES = 32`, `STREAM_ROTATE_FRAMES = 256`, `STREAM_ROTATE_SEC = 2.0`, `DIRECT_RETRY_SEC = 30.0`, `RETRY_MIN_SEC, RETRY_MAX_SEC = 0.5, 10.0`, `RELAY_INITIAL_CREDITS = 16`.
5. **New module `colosseum.hub.relay`** (T3.2): `RelayHub(enabled=, pair_frames=, pair_bytes=, put_timeout_sec=, clock=)` (`set_online`, `stats`, `number_of`, `put`, `attach`, `detach`, `add_credits`, `push_control`, `next_item`), `InboxHandle`, `LatestRelay(root, relay, wait_sec=, make_pull=, clock=)` (`publish_versions`, `latest`, `open_newer`, `complete(..., teacher_active=)`), `RelayedVersion`; constants `RELAY_PAIR_FRAMES = 64`, `RELAY_PAIR_BYTES = 64 MiB`, `RELAY_PUT_TIMEOUT_SEC = 10.0`, `RELAY_PULL_WAIT_SEC = 10.0`. `build_server(..., relay: RelayHub | None = None, latest: LatestRelay | None = None)`.
6. **`colosseum.hub.state` additions:** `version_source(info, entry)`, `SnapshotHolds(refresh_sec, factor=EVICTED_HOLD_FACTOR)` (`grant`, `drop_cell`, `drop_node`, `referenced`, `hold`, `is_held`, `held`, `due`, `max_hold_sec`), `snapshot_refs(lineup)` (T3.1); `collect_allowed(node, learner_node, *, relay, path)`, `strip_collect(lineup, allowed)`, `DROP_REASONS` (T3.2).
7. **`CheckpointManager` additions** (T3.1): `EVICTED_DIR = ".evicted"`, `set_eviction_hold(hold)`, `held_dir(agent, ckpt)`, `release_held(agent, ckpt)`, `clear_held()`; `_evict` keeps a hard-linked copy before deleting when the hold says so. The overview's run-dir table already lists `checkpoints/.evicted/<agent>/ckpt_v<N>/`.
8. **`BlobStore` additions** (T3.1): `frame_path("snapshot", ...)` falls back to `checkpoints/.evicted/`; `forget(kind, agent_id, ident)` (thread-safe; a conversion finishing after `forget` deletes its output).
9. **`NodeOptions`:** `data_addresses: tuple[str, ...] | None = None` (announce these instead of the probed ones: port forwards, public addresses, tests); `data_port == -1` means "the hub config's `distributed.data_port`" (`join`'s default; T6.3 passes it), `0` an ephemeral port, `None` no data port (unchanged).
10. **`NodeAgent` accessors** (published copies, thread-safe for tests and admin): `counters() -> dict[str, int]`, `fetch_history() -> dict[str, list[int]]` (T3.1), `paths() -> dict[int, str]`, `data_port() -> int | None` (T3.2).
11. **`WeightCacheReader.open_latest(agent_id) -> tuple[BinaryIO, dict] | None`** (T3.1 adds to T1.5's module): the current latest segment opened read-only with its pointer dict; used by `GetLatest` and the relay upload to stream a frame without copying it.
12. **Proto additions** (append with the next free field numbers; `PROTOCOL_VERSION` stays `(1, 0)`: nothing is released before the SP5 merge): `Heartbeat.counters` (`map<string, int64>`, node-cumulative per session), `message DataAddresses { repeated string data_addresses; }` as a `NodeMessage` oneof `addresses` (the data port is bound after `Welcome`, when the node knows `distributed.data_port`; `Hello.data_addresses` carries them on every reconnect), `Welcome.banned_node_ids` / `Announce.banned_node_ids` (`repeated string`; data ports refuse these; T6.3 fills them) (T3.1); `message PathState { int32 node; string path; }`, `Heartbeat.paths` (T3.2).
13. **`Download(kind="latest")` and `Upload(kind="relay_latest")`** are served by `LatestRelay` inside the servicer threads; a `relay_latest` upload posts no `UploadIn` event (no hub state is touched; the overview's "UploadIn for every upload" holds for checkpoint / final / recovery uploads). `RelayEvent` is not used: relay counters are read from `RelayHub.stats()` on the metrics tick (posting one inbox event per frame would load the main loop).
14. **`ChunkIntake.note_policy_version(version)`**, `ConnectionIntake(..., max_policy_lag=None)`, `QueueIntake(..., max_policy_lag=None)`, `colosseum.learner.learner.lag_metrics(current, versions)` (T3.3). The stale check runs on the frame header before decoding.
15. **`server_options` must contain `("grpc.so_reuseport", 0)`** (T1.3): grpc-python sets `SO_REUSEPORT` by default, so a second server binds a taken port and the kernel splits connections between the two (verified). Needed by the data port's next-free-port rule and by two hubs on one machine.
16. **`send_parts` accepts any bytes-like object** (a `memoryview` of an `mmap`), not only `bytes | Path` (T2.3): `GetLatest`, relayed `Download` and the relay upload stream from shared memory.
17. **Where the counters live:** node-side and hub-relay counters in the hub's `CounterBook` (`Hub._book`, saved as `hub_state.json` `counters["book"]`, with the contract keys `lineup_fetch_failed`, `relay_bytes`, `dropped_chunks` filled from it); learner-side chunk drops in `MetricsHub.intake_book`. T7.1 sums both into `system.dropped_chunks`; T4.3 restores both on `--resume` (`CounterBook.from_state`).
18. **`train` keys:** `policy_lag_mean` and `policy_lag_max` keep their SP1 meaning (per batch); `policy_lag_p50`, `policy_lag_p95`, `dropped_stale_chunks` (cumulative per learner incarnation) are added.
19. **`LearnerConfig` validator** (T3.3 edits `core/config.py`, owned by T1.4): `max_policy_lag < weight_push_interval - 1` is a `ConfigError` (the learner would starve).
20. **T3.3 also touches `node/learner_main.py`** (one keyword) and `core/config.py`; the overview's index lists `learner.py`, `net/local.py`, `metrics/hub.py`. T3.2 touches neither file, so the T3.2 ‖ T3.3 split holds.
21. **`tests/fault_proxy.py` is created by T3.1** (the overview says T3.2): T3.1's node-fetch test needs it. `TcpFaultProxy(target=None)` with `set_target` (a node announces the proxy's address before its data port is bound), `delay`, `cut`, `drop`, `stall`, `restore`, `connections_seen`, `bytes_forwarded`.
22. **Counter names** (all cumulative, `CounterBook` names): `lineup_fetch_failed`, `snapshot_holds`, `weight_fetches`, `weight_fetch_bytes`, `weight_fetch_failed`, `weight_fetch_not_modified`, `weight_fetch_redundant`, `data_port.latest_served`, `data_port.latest_not_modified`, `data_port.peer_refused`, `node_messages.invalid` (T3.1); `chunks_sent.direct`, `chunks_sent.relay`, `chunks_delivered`, `relay_frames`, `relay_bytes`, `relay_frames_received`, `relay_uploads`, `relay_uploads_refused`, `dropped_chunks.{disconnect,no_learner,invalid,overflow}` (T3.2); `dropped_chunks.{stale,stale_generation,invalid,disconnect}` learner-side (T3.3, `intake_book`).

### Proposed rulings (what — why — cost if wrong)

1. **A broken chunk stream counts all its frames as lost (`disconnect`), an upper bound** — gRPC client streams acknowledge nothing per message; exact accounting would need a bidirectional protocol with acks; streams rotate every 256 frames / 2 s, so a break over-counts by at most what arrived in that stream (scratch run: 5–6 of 900 in a 1.5 s cut) — cost: `dropped_chunks.disconnect` slightly high after a network cut; "sent = received + counted" holds as "≤".
2. **A node fetches only the newest announced version of an agent, at most once per `weight_sync_interval_sec`, and the first weights at once** — spec 4.6 "раз в weight_sync_interval_sec сверяет версии"; intermediate versions are never needed by a worker — cost: none (a worker's lag is bounded by the sync interval, as on one machine).
3. **Agents not named by a node's lineups are fetched once (first weights) and then not refreshed** — saves bandwidth for runs with many agents; the first-weights wait still holds — cost: a lineup naming such an agent plays its stale weights for up to one sync interval after the grant.
4. **Path choice: re-probe the direct path every 30 s while on the relay; without the relay, probe on every retry (0.5 → 10 s backoff)** — spec 4.8 says only "не удалось за connect_timeout_sec — через хаб" — cost: after a short cut the data stays relayed up to 30 s (hub traffic).
5. **Relay drops: `overflow` only after 10 s without room in the pair buffer; frames to an offline node `disconnect` at once** — a slow but live receiver must backpressure, not lose data (spec 4.7); a stuck receiver must not block the sender for ever — cost: a receiver stalled > 10 s loses frames (counted).
6. **Relayed weights are pulled on demand, once per version, and a download waits ≤ 10 s** — pushing every version of a relay-only learner to the hub costs up to model size × push rate; pull-on-demand costs one upload per version per hub — cost: the first relayed fetch of a version is up to one round trip slower.
7. **A node opens `RelayInbox` only while it runs a learner and the relay is on** — no stream for nodes that never receive — cost: none.
8. **The data port is bound after the first `Welcome` and announced with `DataAddresses`** — only then does the node know `distributed.data_port` and the TLS mode — cost: one extra message; until it arrives (milliseconds) the node's learners look relay-only.
9. **`checkpoints/.evicted/` is cleared at hub start** — no granted lineup survives a hub restart that needs it (a node whose grant was pending re-requests) — cost: a pending grant across a `--resume` may count one `lineup_fetch_failed`.
10. **With `distributed.relay: false`, a node without a direct path to an agent's learner gets that agent's seats without collection** (they still play), and its fetches of that agent's latest weights fail (counted, WARNING) — spec 4.8 "недостижимая машина получает только роли без входящих данных" — cost: such lineups wait for weights they cannot get (`lineup_fetch_failed` grows); the WARNING says why.
11. **Policy-lag percentiles are per trained batch** (linear interpolation; `policy_lag_mean` / `policy_lag_max` keep their SP1 meaning) — the `train` record is per step — cost: p95 of a small batch is close to its max.
12. **The stale check runs on the frame header before decoding** (a stale frame is never decoded, so its body is not validated) — saves the decode of a frame that is dropped anyway — cost: a malformed stale frame is counted `stale`, not `invalid`.
13. **`max_policy_lag < weight_push_interval − 1` is a `ConfigError`** — the learner would starve for ever (no new chunks are admissible, no new push comes) — cost: users must raise one of the two keys together.
14. **The data port serves `GetLatest` only for agents whose learner runs on that node** (no re-serving of fetched weights) — one owner per version, no tree distribution in SP5 — cost: 20 nodes download from one learner node per version (spec 8 risk table: accepted, measured in T7.4).
15. **An insecure data port requires the token secret in metadata** — spec 3.3: `--insecure` keeps the token — cost: none.
16. **`tests/fault_proxy.py` moves to T3.1** — its first user is T3.1 — cost: none.

### Assumptions about other parts

1. **T1.1 / T1.2 / T1.3 / T1.5:** names as in the overview; `peek_meta` reads only the header of chunk and weights frames (an lz4 frame's header is plain); `encode_frame(tree, kind=, meta=)` is public (the router test builds header-only frames with it); `receive_parts` raises a `ValueError` subclass for malformed parts; `net_kit` can make registered node identities of one hub (`registered_node(tmp_path, hub, name)` or equivalent); `node.weights` writes the pointer fields `segment`, `version`, `generation`, `size`, `digest`, `teacher_active` and its digest is blake2b-16 of the frame.
2. **T2.3:** `BlobStore` converts `checkpoints/<agent>/<ckpt>/model.pt` lazily into `runs/<name>/blobs/snapshot-<agent>-<ckpt>.frame` and serves `fixed` / `teacher` / `init` frames registered at start; `Download` dispatches on `BlobRequest.kind`; the servicers know the caller's node id from the `AuthInterceptor`; the server's thread pool can be sized for sessions + relay streams (Step 11 of T3.2); `hub_kit` has fake nodes that can open arbitrary streams.
3. **T2.5:** `Hub` can run off the main thread (signal handlers only in the main thread; else T3.1's `cluster_kit` step adds the guard); `HubOptions(port=0)` binds an ephemeral port and `hub.json` lists the bound address; in `hub` mode every trainable agent's learner is placed on the first node with a learner slot (until T4.1); the interim collect rule grants non-collecting lineups to a node without the learner (else T3.1 adjusts it, Step 9 note); `_on_version`, `_on_heartbeat`, `_on_lineup_request`, `_on_snapshot_evicted`, `_flush`-style announcement sending, `hub_state.json` on every metrics tick exist under the R5 names; `_on_metrics_batch` passes each `IntakeStats` to one `MetricsHub` method.
4. **T2.4:** `NodeAgent` has an inbox with a wake pipe, a transfer pool, the lineup-after-cache machinery with a failed-fetch path that drops the grant, counts `lineup_fetch_failed` and re-requests; it computes the `CacheKey`s of every live and pending lineup; it accepts the workers' `route` connections and opens a `chunks` connection to each local learner incarnation (T3.2 Step 7 opens it if not); `FakeHub` can script `Welcome`, grants and announcements and records heartbeats and results; `NodeAgent.run()` can run off the main thread.
5. **T4.1:** generations become real and the intake's generation rule fences stale chunks (`stale_generation`); placement should prefer nodes reachable from the collecting nodes when `distributed.relay` is false (it can read the hub's `_paths`); `learner_node(agent)` comes from the assignment it keeps.
6. **T4.3:** `--resume` restores `Hub._book` from `hub_state.json` `counters["book"]` and `MetricsHub.intake_book` (where T4.3 saves it), and clears `checkpoints/.evicted/` (T3.1 does it at every start anyway).
7. **T4.4:** `hub_lost_collect_sec` uses the router's per-target path (`paths()`: "relay" means the learner is reachable only through the hub); without the hub, `RelayReceiver` and relayed fetches simply fail and retry; direct `GetLatest` polls are T3.1's.
8. **T6.3:** `--ban` fills `Welcome.banned_node_ids` / `Announce.banned_node_ids`; `join` passes `NodeOptions.data_port = -1` (the hub config's port).
9. **T7.1:** prints `system.dropped_chunks` (hub book + `MetricsHub.chunk_drops()`), `relay_bytes`, `lineup_fetch_failed` and `by_node.data_path` (from `Heartbeat.paths`); adds the `train` lag keys to `REQUIRED_KEYS` if it wants them required.
10. **T7.4:** measures criterion 4 with `scripts/measure_weight_cache.py` (T1.5) plus the per-machine fetch counter of this part (`weight_fetches` per node versus distinct versions).
