# SP5: Distributed Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** training runs on several machines as easily as on one:
- a separate **hub** process owns the run (league, journal, ratings, snapshot storage, pulse and tournaments, metrics, budget, stop) and one listening port;
- one **node** agent per machine runs the workers, learners and evaluators the hub assigns, keeps a per-machine shared-memory weight cache, routes chunks between machines (directly, or relayed by the hub) and batches results to the hub;
- a learner of any agent runs on any node; trajectories never pass through the hub's Python objects (relay forwards frames undecoded);
- machines join with one command (`colosseum join`), a curl one-liner, or (SP5b-movable) SSH from the hub machine; the hub ships the code and uv builds the environment;
- TLS and a token are on by default; worker / learner / machine loss and a hub restart are survived;
- every SP3/SP4 feature works across machines (`check_distributed_scope` goes away).

On one machine `colosseum train` stays one command with today's behaviour and ≥ 95 % of `main` `d61eb02`'s throughput; SP3/SP4 configs and run dirs keep working.

**Architecture:**
- **Packages.** New `colosseum.net` (wire layer: array frames, chunk frames and validation, protobuf v1, converters, security, gRPC helpers, same-machine connections), `colosseum.hub` (the hub process: pure state, gRPC servicers, placement, evaluation, admin, registry), `colosseum.node` (node agent, session client, child supervision, shm weight cache, chunk router, child entry points, hardware, logs) and `colosseum.bootstrap` (code bundle, node requirements, torch wheel choice, installer script, thin `join` loader). Removed by the end: `launcher.py` (T2.5, becomes `colosseum.hub.hub`), `distributed.py`, `transport/`, `weight_store/`, `proto/colosseum.proto`, the old `deployment/` files.
- **Hub process** (`colosseum hub`, and the `colosseum train` process): one main thread runs an event loop fed by an inbox (`queue.SimpleQueue`); gRPC servicer threads only parse, authenticate, and put events on the inbox, and stream replies from per-session outbound queues; blob downloads are served from files by servicer threads (read-only). SP4's journal writer and live-ratings threads stay. The hub never spawns workers or learners after T2.5 (until T5.1 it still spawns the SP4 pulse child, ruling R6).
- **Node process** (`colosseum join`, and `train`'s local node = a child of the hub): main thread = a `multiprocessing.connection.wait` loop over its children's connections plus a wake pipe; a session-reader thread, data-port server threads, one sender thread per remote target, a small transfer pool for blobs. Children (workers, learners, evaluators; spawn) die with the node.
- **Same-machine IPC (R3):** per-connection `multiprocessing.connection` Unix sockets with a per-node authkey; control messages are pickled dataclasses; chunks travel as frames (`send_bytes`); weights only through the shm cache (R10). No `mp.Queue`, `mp.Value` or `mp.Event` is shared between restartable processes.
- **Cross-machine IPC:** typed protobuf on gRPC (one session stream per node; blobs in parts; chunk streams to node data ports or through the hub relay). Pickle never crosses machines.
- **Migration (plan-design §3):** libraries first (group A), then the hub server and node agent tested with fakes, then one switch task T2.5 (`train` = `Hub` + local node; `Launcher` deleted, its tests ported); everything after T2.5 is built on the single path and tested on localhost.

**Tech Stack:** Python 3.12 (`requires-python >= 3.11` stays), PyTorch ≥ 2.6, numpy, pydantic v2, click, lz4, `grpcio` + `protobuf` (now runtime dependencies), **`cryptography` (new runtime dependency)**, stdlib `multiprocessing.connection` / `mmap` / `selectors` / `ssl` / `sqlite3` / `http.server` / `tarfile` / `hashlib`, pytest (+pytest-timeout), ruff, uv (also the node-side environment builder; not a Python dependency), `grpcio-tools` (dev only; generated code is committed). Docker is optional (compose tests are `-m docker`).

**Spec:** `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; binding authority; approved by the owner 2026-10-18). Read it fully before any task. Controller decisions for the plan: `.superpowers/sdd/2026-10-17-sp5-distributed/plan-design.md` (rulings R3–R14) and the ledger `progress.md` (rulings R1, R2). Research behind it: `research/sp5_distributed_architectures.md`, `research/sp5_data_plane.md`; repo survey: `.superpowers/sdd/2026-10-17-sp5-distributed/dossier-repo.md`.

Earlier records:
- SP4: spec `docs/superpowers/specs/2026-10-16-sp4-selection-design.md`, plan `docs/superpowers/plans/2026-10-16-sp4-selection/`, acceptance report `docs/superpowers/reports/2026-10-17-sp4-acceptance.md` (rulings in «Решения контроллера во время SP4»).
- SP3: `docs/superpowers/reports/2026-10-10-sp3-acceptance.md`; SP2/SP1 rulings at the ends of `docs/superpowers/reports/2026-10-09-sp2-acceptance.md` and `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; GPU checks `docs/superpowers/reports/2026-10-15-gpu-checks.md`.

## Plan files

| File | Spec blocks | Tasks |
|---|---|---|
| `00-overview.md` (this file) | constraints, file map, interface contract, task index | — |
| `01-foundations.md` | §6 contract tests (P0); 4.9 frame format; 3 security; 9.1 config; 4.6 shm cache library | T0.1, T1.1–T1.5 |
| `02-hub-node.md` | 1; 4.1–4.4; 4.7 (same machine); 5.1; the switch | T2.1–T2.5 |
| `03-weights-data.md` | 4.5–4.8 | T3.1–T3.3 |
| `04-learners-failures.md` | 5.2; 5.4; 7 | T4.1–T4.4 |
| `05-evaluation.md` | 5.3 | T5.1–T5.2 |
| `06-joining.md` | 2 (except SSH); 3.5 | T6.1–T6.3 |
| `07-metrics-docs.md` | 8; 4.10; 9.2 (`validate`); 10 (Docker); 11; measurements; slow tests; acceptance runbook | T7.1–T7.6 |
| `08-sp5b.md` | 2.2 (SSH); 6; 10 (K8s) — may move to SP5b whole (R14) | T8.1–T8.4 |

## Global Constraints

These apply to every task.

- **Dependencies:**
  - runtime: `grpcio` and `protobuf` move from the `grpc` extra into `[project].dependencies` (T1.2; floors = what the generated files check: `GRPC_GENERATED_VERSION` in `_pb2_grpc.py` and the protobuf gencode version in `_pb2.py`); `cryptography>=42` is added (T1.3). The `grpc` extra is removed; `grpcio-tools` moves to the `dev` extra; `scripts/setup-dev.sh` installs `.[dev,examples]` (T1.2).
  - The controller installs `cryptography` into the shared `.venv` once before group A (`uv pip install --python .venv/bin/python 'cryptography>=42'`), so parallel worktrees share it. Implementers never reinstall torch and never install into system Python.
  - `uv` is a node-side tool (installed by the installer script when missing); never a Python dependency. No Ray, Redis, `safetensors`, K8s client or new web/JS libraries.
- **Environment (owner's machine: WSL2, ROS):** prefix every `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff` command with `env -u PYTHONPATH`; parallel worktrees use the main checkout's venv with `PYTHONPATH=<worktree>/src` (process README). WSL2 networking is NAT: inbound connections from the LAN do not reach it (owner's machine setting; only acceptance needs it).
- **Clocks:** every timeout, lease, TTL, cadence and duration uses `time.monotonic()`; the hub measures node loss and block leases on its clock, a node measures its cells' lineup TTLs on its clock; clocks of different machines are never compared and nothing crosses the wire as a deadline (only durations: `ttl_sec`, `lease_sec`). Order data by counters (`env_steps`, versions, epochs, batch seqs, journal row ids), never by `ts`. Tests never assert on wall-clock values or rates.
- **Backward compatibility:**
  - every config under `configs/examples/` and `tests/fixtures/sp{2,3}/configs/` validates and trains on one machine without edits (`transport.*` keys are translated with a WARNING, T1.4);
  - an SP4 run dir (T0.1 fixture `tests/fixtures/sp4/`) resumes in place (`train --resume`, T4.3) and as a child run (`training.resume_from`, unchanged semantics);
  - SP3 run dirs and SP2/SP3 checkpoints keep their SP4 behaviour (resume, frozen agents, `init`, tournaments);
  - `MatchResult(...)`, `SeatResult(...)`, `Lineup(...)`, `WeightPayload(...)` built without the new fields keep working (defaults);
  - two `colosseum train` on one machine run side by side (ephemeral loopback ports, run-scoped shm and socket names).
- **Green suite after every task:** the task's new tests pass; `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` passes in full **with zero warnings**; `env -u PYTHONPATH .venv/bin/ruff check .` passes.
  - Tasks that change the learner, the worker, the training path or devices (T2.2, T2.5, T3.3, T4.1, T4.2, T7.4) run the slow suite once (`pytest -m slow -v`, alone on a quiet machine, in the background); T2.5 and T4.1 also run `pytest -m gpu -q -W error` (CUDA is present).
  - Measurements (T1.1, T1.5, T2.5, T7.4) run only on a quiet machine, one at a time; their thresholds are fixed in the task text before measuring.
- **Inter-process data (R3):**
  - same machine (node ↔ its children): `multiprocessing.connection` with the node's authkey; control messages are pickled dataclasses of `colosseum.net.local`; chunk and checkpoint payloads are frames (`send_bytes`); weights only via the shm cache;
  - across machines: protobuf messages and frames only; **no pickle**, no `torch.save` bytes decoded by the receiver except a learner's own trainer state, which is read with `torch.load(..., weights_only=True)`;
  - nothing that crosses a process boundary contains a `torch.Tensor` (spawn kwargs included).
- **No shared locks between restartable processes (spec principle):** every child talks to its receivers over its own connections; each learner incarnation listens on its own address; env-step counters are cumulative per worker incarnation; stop is a message. A `kill -9` of any child must not block its node or its siblings (tested in T2.2, T4.2).
- **Threads in the hub:** the main loop owns every piece of hub state (coordinator, storage, registry, cells, budget, versions, evaluation queue); servicer threads only parse, authenticate, put inbox events and read files; the journal writer and live-ratings threads keep their SP4 rules (they never call `Coordinator` / `CheckpointManager` / `PfspStats` / `RatingBook`); a hub-run tournament drives its modes in a thread that talks to the main loop only through inbox events and futures (T5.2). The main loop never blocks on the network or on a node.
- **Threads in the node:** the main loop owns children, cells, the results buffer and the cache index; the session reader, data-port servicers, sender threads and transfer pool only post into the node inbox (`queue.SimpleQueue` + wake pipe).
- **Errors:**
  - config problems: `ConfigError` with a fix hint; env contract violations `EnvContractError`, bot problems `PlayerError` — their context gains the node (`node N (name), worker W, env E, seat P, episode step K, layout L, agent A`) and the path of the log (local and, for remote nodes, the hub's `.log.tail`);
  - anything a node sends is untrusted: a malformed frame, chunk, result or announcement is dropped, counted by reason, with one WARNING per node and reason; it never stops the hub, a learner or a node;
  - no silent fallbacks beyond the spec's and the inherited ones; every lost chunk, result or block result is counted.
- **Security defaults:** TLS on the hub port and mutual TLS on node data ports unless `--insecure`; the token is required in every mode (also loopback and `--insecure`); private keys and secrets are written `0600`, identity dirs `0700`; the dashboard binds `127.0.0.1` by default.
- **Tests:**
  - new test files use the prefix `test_sp5_`; basenames and support-module names unique across `tests/`; no `__init__.py`; support modules are imported by bare name and listed in `known-first-party` of `pyproject.toml` by the task that creates them;
  - files only under `tmp_path`: the autouse fixture `_isolate_colosseum_dirs` (T1.3, `tests/conftest.py`) redirects `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `COLOSSEUM_SHM_DIR` into `tmp_path` for every test (children inherit them through `cli_runner.child_env`); two nodes of one test get two identity roots (`NodeOptions.identity_root`, or per-process env);
  - at most **2 loaded CPU processes per fast test** (workers, evaluators, learners, tournament workers; hub and node processes, mostly idle, are not counted), at most 3 per slow test; `OMP_NUM_THREADS=1`;
  - hub logic is tested on `colosseum.hub.state` objects and with fake nodes (in-process clients of the real gRPC server on loopback); node logic with a fake hub (an in-process server) (R13); real processes only where the process boundary is the subject;
  - processes are started through `tests/cli_runner.py` (process-group cleanup; `run_in_session` for `hub`, `join`, `tournament --hub`); ports come from the code under test binding port 0 (no bind-then-release races);
  - `@pytest.mark.docker` (new marker, T7.3) for tests that need Docker; they skip without it.
- **SP1–SP4 rulings stay in force** unless spec §8 changes them. Most likely to be broken by accident:
  - the local 10 s stop guarantee (`SHUTDOWN_GRACE_SEC = 7`, no process alive after 10 s) for `train` on one machine; the distributed stop uses `distributed.shutdown_grace_sec`;
  - exactly `learner.batch_chunks` chunks per update; `training.total_timesteps` is the global env-step budget, and the hub is its only authority;
  - the legality gate, `EnvContractError` / `PlayerError` context formats, play-scope validation of `record` / `bc` / `eval` / `tournament`;
  - the first-weights wait (A-M3: no match before every playable agent's first weights);
  - SP4's journal, retention, ratings, pulse and tournament semantics (`RUN_SOURCES`, `MatchLog` dedup by uid, put-then-evict R-C1, the pre-refit 2 × cap, pulse never evaluating a local run's final snapshot, `pulse_state.json` not resumed);
  - PFSP statistics of evicted snapshots are never revived; retention runs on the main loop only;
  - changed by the spec (§8) and therefore not preserved: the missing-snapshot fallback to latest + collect (gone, R11 / spec 4.2), newest-wins weight mailboxes and shared queues (gone), `--resume` semantics (in place; `training.resume_from` unchanged), the journal schema (`node` column).
- **Config:** the `distributed` section is global only (like SP4's sections); `learner.*` keys keep their per-agent override; **any config-schema change also updates `scripts/bench_throughput.py::_make_config`** (guarded by `tests/unit/test_bench_throughput.py`).
- **Language:** code, comments, docstrings, log messages, CLI help, report/dashboard UI text, `README.md`, `CLAUDE.md` in English; `docs/DISTRIBUTED_GUIDE.md`, `docs/ARENA_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, `docs/benchmarks.md` and the acceptance report in Russian.
- **Commits:** conventional prefixes; no attribution or co-author lines; at least one commit per task on branch `sp5-distributed` (or its worktree branch); push after every task.
- **Style:** follow existing code (type hints, dataclasses, pydantic, `logger = logging.getLogger(__name__)`, `from __future__ import annotations`, module docstrings that state the contract).
- **Process (owner's instruction):** one Opus/high implementer per task (TDD, commit, push; no subagents of its own), one Opus/high reviewer per task; web research by Sonnet/high subagents.

## Deviations from the spec

- **Numbering:** spec §7 parts P0–P8 map to task groups T0–T8 (P0 → T0.1, P1 → T1.x, …, P8 → T8.x). T1.4 (config), T1.5 (shm cache library) and T6.1 (bundle libraries) are moved forward into group A because they are pure libraries.
- **R1 — one port, application-level node authentication.** Spec 3.2 says every connection is mutual TLS. grpc-python cannot request an optional client certificate, and the hub port must also serve certificate-less clients (installer, first registration). So: the hub port is server-side TLS (the node pins the CA from the token); every hub-port RPC after registration carries the node id + a per-node secret in metadata (only its SHA-256 is stored by the hub); node data ports use full mutual TLS with hub-issued node certificates. `--ban` / `--rotate` revoke both.
- **The hub's TLS certificate is the CA certificate itself** (contract §Security; Question Q1): a separate server certificate signed by the CA (spec 2.1) gives the node no portable way to obtain the CA certificate before it trusts the hub (Python 3.12's `ssl` exposes only the leaf). With the CA as the server certificate the node fetches it unverified (`ssl.get_server_certificate`), checks its SPKI fingerprint against the token, then uses it as the gRPC root; the curl one-liner pins the same key. Node certificates are still issued by the CA.
- **Stale generations are fenced by a version cap** (Question Q2): results and chunks of a superseded learner generation `g` are rejected only above `cap(g)` (the highest version the hub accepted from `g` before it assigned `g + 1`); versions up to the cap were announced, are unique, and stay valid. Spec 7.2 rejects "results of seats of a stale generation"; the cap rule rejects exactly the versions that could collide.
- **Epochs reach the runner through `Lineup.epoch`** (Question Q3): spec 4.3 / 7.6 need the epoch in every training match id and episode seed, which `LoopIO` alone cannot carry. `Lineup` gains `epoch`, `MatchRunner` / `RolloutLoop` gain an optional `cell_seed_fn` keyword and `RolloutLoop` an optional `uid_prefix` (the `n<node>/` uid segment); with the defaults both behave exactly as in SP4 (R4's contract tests stay valid).
- **`train`'s local node (R12) keeps today's capacity:** every trainable agent's learner is placed on it (a GPU takes any number of learners, as today; CPU learners are unlimited); evaluators: one slot from T5.1, run on demand (A10); no data port without `--listen`. Spec 2.6's slot defaults apply to `join` nodes.
- **lz4 only for frames that leave the machine.** Spec 4.9's adaptive per-agent compression is applied by senders whose target is another machine (node router, relay, weight downloads stay uncompressed by default); a worker's frame to a learner on its own machine is never compressed (criterion 2).
- **`tournament --hub ADDR|RUN_DIR`:** `tournament` already uses `--run` for candidate runs, so the spec 3.3 rule "`--run RUN` names the hub" is spelled `--hub RUN_DIR` there (a directory means "read its `hub.json`").
- **`hub token --rotate` refuses while a hub of this identity runs** (stop it first); spec 2.1 does not say what a running hub does with a rotated CA.
- **Node result buffer overflow drops the newest results** (the buffer keeps resending the oldest); spec 4.3 only says overflow is counted.

---

## File structure

### Create
- `src/colosseum/net/__init__.py` — package docstring (T1.1).
- `src/colosseum/net/frame.py` — array frame codec, hostile-input checks, `Lz4Probe`, `peek_meta` (T1.1).
- `src/colosseum/net/chunks.py` — chunk / weights / checkpoint frames, `ChunkValidator` (T1.1).
- `src/colosseum/net/proto/__init__.py`, `colosseum_v1.proto`, generated `colosseum_v1_pb2.py`, `colosseum_v1_pb2_grpc.py`; `scripts/gen_proto.sh` (T1.2).
- `src/colosseum/net/messages.py` — `MatchResult` / `Lineup` / `SeatAssignment` / version converters, `PROTOCOL_VERSION` (T1.2).
- `src/colosseum/net/security.py` — identities, token, certificates, `NodeBook`, credentials, auth (T1.3).
- `src/colosseum/net/grpc_util.py` — channel/server options, metadata, `AuthInterceptor` (T1.3); blob streaming in parts (T2.3).
- `src/colosseum/net/local.py` — connections, addresses, control messages, worker/learner IO (T2.2).
- `src/colosseum/hub/__init__.py`, `hub/state.py` (T2.1); `hub/server.py` (T2.3); `hub/hub.py`, `hub/registry.py`, `hub/admin.py` (T2.5); `hub/placement.py` (T4.1); `hub/evaluation.py` (T5.1).
- `src/colosseum/node/__init__.py`, `node/weights.py` (T1.5); `node/session.py` (T2.3); `node/agent.py`, `node/procs.py`, `node/worker_main.py`, `node/learner_main.py`, `node/hardware.py`, `node/logs.py` (T2.4); `node/router.py` (T3.2); `node/evaluator_main.py` (T5.1).
- `src/colosseum/bootstrap/__init__.py`, `bundle.py`, `requirements.py`, `torch_select.py` (T6.1); `installer.py`, `loader.py` (T6.2).
- `scripts/measure_frame_decode.py` (T1.1), `scripts/measure_weight_cache.py` (T1.5), `scripts/measure_hub_load.py`, `scripts/measure_scaling.py` (T7.4).
- `tests/fixtures/sp4/` — SP4 run-dir fixture + `make_sp4_run.py` + `manifest.json` (T0.1).
- Test support: `tests/net_kit.py` (T1.3), `tests/hub_kit.py` (T2.3), `tests/node_kit.py` (T2.4), `tests/cluster_kit.py`, `tests/fault_proxy.py` (T3.1), `tests/eval_kit.py` (T5.1), `tests/join_kit.py` (T6.2), `tests/ssh_kit.py` (T8.1) (A4, A36).
- `deployment/Dockerfile.node`, `deployment/compose.yaml`, `deployment/compose.node.yaml` (T7.3); `deployment/k8s/{hub,node-cpu,node-gpu,token-secret}.yaml` (T8.4).
- `docs/DISTRIBUTED_GUIDE.md` (T7.5).

### Modify (main ones)
- `src/colosseum/core/types.py`: `Lineup.epoch`, `SeatResult.generation`, `MatchResult.node` (T1.2); `WeightPayload.generation` (T2.2).
- `src/colosseum/core/config.py`: `DistributedConfig`, `LearnerPlacementConfig`, `LearnerConfig.{max_policy_lag, recovery_interval_sec, placement}`, `transport` translation, `TransportConfig` / `TransportMode` removed (T1.4).
- `src/colosseum/core/validation.py`: `_check_state_payload` on the frame codec (T1.1); `distributed:` line and placement plan (T7.2).
- `src/colosseum/worker/match_runner.py`, `worker/rollout_loop.py`: epoch-aware match ids and seeds, `cell_seed_fn` (T2.2).
- `src/colosseum/learner/learner.py`: intake / outputs abstractions, `version_floor`, `generation` (T2.2); lag percentiles, `max_policy_lag` (T3.3); recovery points (T4.2).
- `src/colosseum/utils/seeding.py`: `cell_seed`, `worker_seed` (T2.2).
- `src/colosseum/utils/process.py`: portable parent-death watch and process liveness (T4.4).
- `src/colosseum/core/run_state.py`: portable `process_start_ticks` (T4.4).
- `src/colosseum/coordinator/coordinator.py`: `lineup_for`, `save_league_state(..., journal_row=)` (T2.5); `load_league_state` returns the watermark (T4.3); dynamic membership (T8.2).
- `src/colosseum/arena/matchlog.py`: `AppendStats.rows`, `MatchLogWriter.submit(..., token=)`, `take_commits` (T2.5); `node` column, `SCHEMA_VERSION = 2`, migration (T4.3).
- `src/colosseum/arena/pulse.py`: blocks instead of the pulse child (T5.1); `arena/tournament/run.py`: hub-run driver hooks (T5.2).
- `src/colosseum/metrics/{aggregator,hub,jsonl,console}.py`: worker stats keyed by (node, worker, incarnation) (T2.5); `system` / `train` keys, nodes line (T7.1, T3.3, T4.1).
- `src/colosseum/arena/report/{data,render,server}.py` + assets: «Nodes» section, policy lag (T7.1).
- `src/colosseum/cli.py`: `train` / `hub` (T2.5), `--resume` (T4.3), `tournament --hub` + `hub --idle` (T5.2), `join` / `nodes list|remove` / `hub token` (T6.3), `hub --dashboard` (T7.1), old roles → exit 2 (T7.2), `--nodes` / `nodes add` (T8.1), `agents …` (T8.2), `nodes update-code` (T8.3).
- `scripts/bench_throughput.py`: `_make_config` (T1.4), `run_one` on `Hub` with a `Launcher` fallback for `--tree` (T2.5).
- `pyproject.toml`: dependencies (T1.2, T1.3), `known-first-party` (each kit task), ruff excludes (T1.2, T7.2), `docker` marker (T7.3); `scripts/setup-dev.sh` (T1.2).
- `tests/conftest.py`: `_isolate_colosseum_dirs` (T1.3); `tests/game_helpers.py`: `SP4_RUN`, `copy_sp4_run` (T0.1); `tests/cli_runner.py`: hub/join helpers (T2.5, T6.3).
- Docs: `README.md`, `CLAUDE.md`, `docs/ARENA_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, `docs/benchmarks.md` (T7.4, T7.5); `docs/superpowers/process/brief.sh` glob (controller).

### Delete
- `src/colosseum/launcher.py` (T2.5; tests ported, the task report lists every ported / deleted test).
- `src/colosseum/distributed.py`, `src/colosseum/transport/` (incl. `serialization.py` — its hostile-input tests are ported to the frame codec), `src/colosseum/weight_store/`, `proto/colosseum.proto`, their tests (`test_sp2_distributed_e2e`, `test_sp2_grpc`, `test_sp2_serialization` (ported), `test_sp2_distributed_roles`, `test_sp3_distributed_guards`, `test_sp4_distributed_guards`, the distributed parts of `test_sp2_payloads`, `test_sp2_learner_entry`, `test_sp3_learner_factory`, `test_sp4_snapshot_store`, `test_sp4_first_weights`, `test_sp2_entry_points`; `test_sp3_dagger_chunks` ported to `ChunkValidator`) (T7.2). `worker.rollout_worker.rollout_worker_process` and unused `core.ipc` helpers go too if nothing else uses them (T7.2 greps; `ResultBatcher` stays).
- `deployment/Dockerfile.base`, `Dockerfile.weight-store`, `Dockerfile.worker`, `docker-compose.yaml`, `k8s/*` (T7.3; T8.4 adds the new manifests).

---

## Interface contract

Names that cross task or part boundaries. Each part may add names in its own `## Contract notes`; conflicts are reconciled into «Contract amendments» below before execution. `(Tn.m)` = the task that creates the item.

### Constants (shared)

```python
# colosseum.net.messages (T1.2)
PROTOCOL_VERSION = (1, 0)            # (major, minor): a major mismatch is refused at Bootstrap/Info and Hello
# colosseum.hub.state (T2.1)
EPOCH_BLOCK = 1024                   # R7: epochs reserved per hub_state.json write
EVICTED_HOLD_FACTOR = 2              # spec 4.2: an evicted snapshot stays servable <= 2 x match_refresh_interval_sec
# colosseum.hub.hub (T2.5)
INBOX_POLL_SEC = 0.05                # main loop wait on the inbox; periodic work runs on monotonic deadlines
LOCAL_SHUTDOWN_GRACE_SEC = SHUTDOWN_GRACE_SEC   # 7.0: train on one machine (10 s guarantee)
# colosseum.node.session (T2.3)
RECONNECT_MIN_SEC, RECONNECT_MAX_SEC = 0.5, 10.0          # spec 4.1 backoff (doubling, jitter <= 20 %)
HUB_LOST_RETRY_SEC = (30.0, 60.0)    # spec 7.1: retry pause after hub_lost_timeout_sec
# colosseum.node.procs (T2.4)
WORKER_RESTARTS = (3, 600.0)         # max restarts per window (s): workers and evaluators (spec 7.1)
LEARNER_RESTARTS = (3, 1800.0)       # learners
# colosseum.node.logs (T2.4)
LOG_TAIL_BYTES = 64 * 1024
# colosseum.net.frame (T1.1)
FRAME_MAGIC = b"CSTF"; FRAME_VERSION = 1; FRAME_ALIGN = 64
MAX_HEADER_BYTES = 1 << 20; MAX_LEAVES = 100_000; MAX_DEPTH = 64
DECOMPRESSED_SIZE_FACTOR = 8         # payload_byte_cap(mb) = 8 x mb MiB (SP1 rule)
LZ4_MIN_RATIO = 1.3; LZ4_PROBE_FRAMES = 4; LZ4_REPROBE_SEC = 600.0
```

### `colosseum.net.frame` (T1.1)

Byte layout of every frame (chunks, weights, snapshots, frozen / teacher / init weights, checkpoints, recovery points):

```
offset 0   4 B  magic b"CSTF"
       4   2 B  uint16 LE format version (1)
       6   2 B  uint16 LE flags (bit 0: data region is one lz4 frame)
       8   8 B  uint64 LE header length H (<= MAX_HEADER_BYTES)
      16   H B  UTF-8 JSON header
  align64(16 + H)  data region: raw little-endian buffers, each at a 64-aligned offset (or one lz4 frame of it)
```

Header JSON: `{"v": 1, "kind": "tree"|"chunk"|"weights"|"checkpoint", "meta": {...}, "tree": NODE, "leaves": [{"dtype": "<f4", "shape": [..], "offset": o, "nbytes": n}, ...], "data_bytes": D, "lz4": null | {"raw_bytes": D}}`; `NODE` = `{"t": "nd", "i": k}` (leaf k) | `{"t": "dict", "k": [keys], "v": [NODE]}` (keys str/int/float/bool/null) | `{"t": "list"|"tuple", "v": [NODE]}` | `{"t": "nt", "cls": "module:Qualname", "f": [fields], "v": [NODE]}` | `{"t": "py", "v": scalar}` (None/bool/int/float/str). Offsets are relative to the data region.

Hostile-input rules (decode raises `FrameError(reason)` before allocating): header length ≤ `MAX_HEADER_BYTES`; total declared `nbytes` (and `lz4.raw_bytes`) ≤ the caller's cap; each leaf `nbytes == prod(shape) * itemsize`, dims ≥ 0; offsets 64-aligned, inside `data_bytes`, non-overlapping; dtype whitelist `bool, int8..int64, uint8..uint64, float16, float32, float64` (little-endian or `|`), never `object`; leaf count ≤ `MAX_LEAVES`, depth ≤ `MAX_DEPTH`; a namedtuple class must already be imported (`sys.modules`, no import) and have the same fields; lz4 decompression is bounded (`LZ4FrameDecompressor` with `max_length`); no pickle anywhere.

```python
class FrameError(ValueError):
    reason: str                       # short machine reason ("header_too_large", "overlap", "dtype", ...)
    def __init__(self, reason: str, message: str = "") -> None   # A16
def payload_byte_cap(max_message_mb: int) -> int
def encode_frame(tree: Any, *, kind: str = "tree", meta: Mapping[str, Any] | None = None, compress: bool = False) -> bytes
def decode_frame(buf: bytes | memoryview, *, cap: int, copy: bool = False, kind: str | None = None) -> tuple[Any, dict]   # (tree, meta); zero-copy views unless copy (A16)
def peek_meta(buf: bytes | memoryview) -> tuple[str, dict]          # (kind, meta) from the header only (routing; no arrays)
class Lz4Probe:                        # per (sender, agent): adaptive lz4 (spec 4.9)
    def __init__(self, *, min_ratio: float = LZ4_MIN_RATIO, probe_frames: int = LZ4_PROBE_FRAMES,
                 reprobe_sec: float = LZ4_REPROBE_SEC, clock: Callable[[], float] = time.monotonic) -> None
    def compress(self) -> bool         # use lz4 for the next frame?
    def observe(self, raw_bytes: int, compressed_bytes: int | None) -> None
```

### `colosseum.net.chunks` (T1.1)

```python
@dataclass(frozen=True)
class ChunkMeta:
    agent_id: str
    policy_version: int
    generation: int          # learner generation of the weights with policy_version (0 = unknown / SP4 path)
    node: int                # sender node number (-1 local tests)
    worker: int
def encode_chunk(chunk: TrajectoryChunk, *, generation: int, node: int, worker: int, compress: bool = False) -> bytes
def decode_chunk(buf: bytes | memoryview, *, cap: int, validator: ChunkValidator | None = None) -> tuple[TrajectoryChunk, ChunkMeta]   # A17
class ChunkError(ValueError):
    reason: Literal["invalid", "stale", "stale_generation"]
    def __init__(self, reason: str, detail: str = "") -> None   # A16
class ChunkValidator:
    """Spec 4.9 checks on the learner: validate_slot_structure; obs / global_state / action / mask leaf shapes
    [S, ...] and dtypes against the agent's RoleSpec; finite behavior_logp (and behavior_unit_logp) on ACT slots;
    zero rewards on BOOT / PAD; teacher_action inside the action space and has_teacher bool, both or neither;
    agent_id == the learner's agent."""
    def __init__(self, role_spec: RoleSpec, *, agent_id: str, teacher: bool, chunk_length: int | None = None,
                 state_template: Any = None) -> None                # A17
    def check(self, payload: dict, agent_id: str, policy_version: int | None = None) -> None   # numpy payload (A17)
    def validate(self, chunk: TrajectoryChunk) -> None            # raises ChunkError("invalid")
@dataclass(frozen=True)
class WeightsMeta:
    agent_id: str
    kind: Literal["latest", "snapshot", "fixed", "teacher", "init", "recovery"]
    generation: int
    policy_version: int
    teacher_active: bool = False
    checkpoint_id: str | None = None
    architecture: dict | None = None     # meta.json "networks" (or the frozen entry's networks) for fixed / teacher / init
    roles: tuple[str, ...] | None = None
def encode_weights(state_dict: Mapping[str, np.ndarray], meta: WeightsMeta) -> bytes    # never compressed
def decode_weights(buf: bytes | memoryview, *, cap: int) -> tuple[dict[str, np.ndarray], WeightsMeta]
def encode_checkpoint(payload: dict, *, kind: Literal["periodic", "final", "recovery"], generation: int) -> bytes
    # payload = learner.make_checkpoint_payload(); trainer_state_bytes travel as one uint8 leaf
def decode_checkpoint(buf: bytes | memoryview, *, cap: int) -> tuple[dict, dict]   # (payload as make_checkpoint_payload, meta)
```

### `colosseum.net.proto` and `colosseum.net.messages` (T1.2)

`src/colosseum/net/proto/colosseum_v1.proto`, `package colosseum.v1;` (generated with `-I src`, so imports are `colosseum.net.proto.colosseum_v1_pb2`). Outline (field numbers and exact scalar types are T1.2's; versions and counters are `int64`):

```proto
service Bootstrap {
  rpc Info(Empty) returns (BootstrapInfo);                 // no auth
  rpc Bundle(BundleRequest) returns (stream BlobPart);     // token secret
  // /colosseum.v1.Bootstrap/Script: raw bytes (identity serializers, R2), registered as a generic handler, no auth
}
service Hub {
  rpc Register(RegisterRequest) returns (RegisterReply);           // token secret
  rpc Session(stream NodeMessage) returns (stream HubMessage);     // node id + node secret (all below)
  rpc Upload(stream UploadPart) returns (UploadReply);             // checkpoints, recovery points, final, relayed weights
  rpc Download(BlobRequest) returns (stream BlobPart);             // snapshots, fixed, teacher, init, recovery, relayed latest
  rpc RelaySend(stream ChunkFrame) returns (ChunkSummary);         // sender node -> hub -> target node
  rpc RelayInbox(stream RelayCredit) returns (stream RelayItem);   // receiver node <- hub
  rpc Admin(AdminRequest) returns (AdminReply);                    // token secret
  rpc AdminStream(AdminRequest) returns (stream AdminReply);       // token secret (tournament progress)
}
service NodeData {                                                  // node data port, mutual TLS
  rpc Chunks(stream ChunkFrame) returns (ChunkSummary);
  rpc GetLatest(LatestRequest) returns (stream BlobPart);           // first part: not_modified, or meta of a weights frame
}
message BootstrapInfo { uint32 protocol_major; uint32 protocol_minor; string code_hash; string requirements_hash;
  repeated string requirements; string python; string torch_version; string torch_build; int64 bundle_size; }
message BundleRequest { string code_hash; }
message BlobPart { bytes data; int64 offset; int64 total_size; bool last; string digest; bool not_modified; string meta_json; }
message RegisterRequest { string public_key_pem; string machine_id; string name; repeated string tags; string node_id; }
message RegisterReply { string node_id; int32 number; string name; string certificate_pem; string node_secret; string ca_pem; }
message NodeMessage { oneof m { Hello hello; Heartbeat heartbeat; LineupRequest lineup_request; ResultBatch results;
  MetricsBatch metrics; VersionAnnounce version; Event event; BlockLeaseRequest lease_request; BlockResult block_result; } }
message HubMessage { oneof m { Welcome welcome; LineupGrants grants; Ack ack; Command command; Announce announce;
  BlockLease lease; } }
message Hello { string node_id; string session_id; uint32 protocol_major; uint32 protocol_minor; string code_hash;
  Hardware hardware; Slots slots; repeated string data_addresses; string run_id; repeated LiveCell live_cells;
  int64 env_steps; int64 last_acked_seq; repeated RunningLearner learners; string name; repeated string tags; }
message Hardware { int32 logical_cores; int32 physical_cores; int64 memory_bytes; repeated Gpu gpus; string os;
  string platform; string python; string torch_version; string torch_build; int64 shm_free_bytes; int64 cache_free_bytes; }
message Gpu { int32 index; string name; int64 memory_bytes; string kind; }          // kind: cuda | mps
message Slots { int32 workers; int32 evaluators; int32 cpu_learners; repeated int32 gpus; bool no_learners; int32 envs_per_worker; }
message LiveCell { int32 worker; int32 env; int64 epoch; int32 incarnation; }
message RunningLearner { string agent_id; int64 generation; int64 version; }
message Welcome { int32 node_number; string run_id; string config_json; repeated AgentState agents;
  repeated VersionInfo versions; repeated Eviction evictions; repeated LearnerAssignment learners;
  int32 evaluators; int64 env_steps; repeated FixedBlob fixed; bool collect_allowed; }
message AgentState { string agent_id; string kind; string status; int64 start_env_steps; }   // status: active|pending|failed|frozen|removed
message FixedBlob { string agent_id; string blob_id; string kind; }   // kind: fixed | teacher | init (for the agent)
message Heartbeat { int64 env_steps; int64 dropped_results; repeated ProcState procs; double load; repeated string cached_keys; }
message ProcState { string name; string state; int32 restarts; }
message LineupRequest { repeated CellRequest cells; }
message CellRequest { int32 worker; int32 env; int64 epoch; int32 incarnation; }   // epoch 0 = none yet
message LineupGrants { repeated Grant grants; }
message Grant { int32 worker; int32 env; int64 epoch; double ttl_sec; LineupProto lineup; }   // ttl 0 = whole run
message LineupProto { string layout; repeated SeatProto seats; optional int64 seed; optional string block;
  optional string uid; optional int64 epoch; }
message SeatProto { string agent_id; string network_id; bool collect; string source; }
message ResultBatch { int64 seq; repeated MatchResultProto results; }
message MatchResultProto { string match_id; string layout; string outcome_kind; repeated SeatResultProto seats;
  repeated TeamResultProto teams; int64 episode_length; string uid; string source; optional int64 seed;
  optional int64 env_steps; optional string block; optional int32 node; }
message SeatResultProto { int32 seat; string role; int32 team; string agent_id; string network_id; double reward;
  optional int64 eliminated_step; string source; optional int64 version_start; optional int64 version_end;
  optional int64 generation; }
message TeamResultProto { int32 team; double rank; double score; }
message MetricsBatch { repeated TrainMetrics train; repeated WorkerStats workers; repeated IntakeStats intake; }
message TrainMetrics { string agent_id; int64 generation; int64 train_step; map<string, double> values; }
message WorkerStats { int32 worker; int32 incarnation; map<string, int64> counters; }
message IntakeStats { string agent_id; int64 generation; int32 depth; map<string, int64> dropped; }
message Ack { repeated int64 seqs; }
message VersionAnnounce { VersionInfo info; }
message VersionInfo { string agent_id; int64 generation; int64 version; int64 size; string digest; int32 node;
  repeated string addresses; bool relay_only; bool teacher_active; }
message Event { string kind; string proc; int32 exit_code; string error_kind; string message; string context;
  bytes log_tail; string log_path; }   // kinds and the added fields of every message: A1
message Command { string kind; LearnerAssignment learner; string reason; string code_hash; int32 evaluators; }
  // kind: start_learner | stop_learner | set_evaluators | release | stop_run | update_code | cancel_leases (A1)
message LearnerAssignment { string agent_id; int64 generation; int64 version_floor; string device;
  string weights_blob; string trainer_state_blob; int64 env_steps; }
message Announce { repeated VersionInfo versions; repeated Eviction evictions; repeated AgentState agents;
  int64 env_steps; bool collect_allowed; }
message Eviction { string agent_id; string checkpoint_id; }
message BlobRequest { string kind; string agent_id; string ident; int64 since_version; }
  // kind: snapshot | fixed | teacher | init | latest | start | eval (recovery / trainer_state only until T4.1: A28)
message UploadPart { oneof p { UploadHeader header; bytes data; } }
message UploadHeader { string kind; string agent_id; int64 generation; int64 version; int64 total_size; string digest;
  string request_id; }   // kind: checkpoint | final | recovery | relay_latest | agent_file (A1)
message UploadReply { bool accepted; string reason; string checkpoint_id; }
message ChunkFrame { string agent_id; int32 target_node; int64 generation; bytes frame; }
message ChunkSummary { int64 received; int64 dropped; }
message RelayCredit { int32 frames; string node_id; }
message RelayItem { oneof i { ChunkFrame chunk; LatestRequest pull; } }
message LatestRequest { string agent_id; int64 since_version; string request_id; }
message BlockLeaseRequest { int32 free_slots; }
message BlockLease { string lease_id; double lease_sec; repeated BlockSpec blocks; }
message BlockSpec { string block_id; string source; repeated LineupProto lineups; repeated PlayerRefProto players;
  bool deterministic; }
message PlayerRefProto { string name; string agent_id; string network_id; repeated string roles; string kind;
  string blob_id; string networks_json; string bot_class; string bot_kwargs_json; }
message BlockResult { string lease_id; string block_id; repeated MatchResultProto results; bool failed; string error; }
message AdminRequest { string command; string args_json; }
message AdminReply { int32 code; string text; string data_json; bool done; }
```

`colosseum.net.messages` (T1.2) — the only Python converters (every other session message is used as its protobuf class):

```python
def result_to_proto(r: MatchResult) -> pb.MatchResultProto
def result_from_proto(p: pb.MatchResultProto) -> MatchResult           # every SP4 field + node, seat generation
def results_from_batch(p: pb.ResultBatch) -> list[MatchResult]
def lineup_to_proto(l: Lineup) -> pb.LineupProto
def lineup_from_proto(p: pb.LineupProto) -> Lineup
def version_to_proto(v: VersionInfo) -> pb.VersionInfo; def version_from_proto(p) -> VersionInfo
# VersionInfo (the dataclass of hub.state below, with to_dict / from_dict) and EVENT_KINDS are defined here (A1, A7)
```

### `colosseum.core.types` changes (T1.2; used from T2.2)

```python
@dataclass
class Lineup:                 # existing fields unchanged, then:
    epoch: int | None = None  # the cell's epoch (training lineups from the hub; spec 4.2). With an epoch and no seed:
                              # match_id = f"w{W}_e{E}_g{epoch}_ep{K}", episode seed = SeedSequence([cell seed, K])
@dataclass
class SeatResult:             # existing fields unchanged, then:
    generation: int | None = None   # learner generation of the seat's weights at the episode end (latest seats)
@dataclass
class MatchResult:            # existing fields unchanged, then:
    node: int | None = None   # node number of the worker that played it (None: SP4 rows, tournaments run locally)
```

Training uids become `<run_id>/train/n<node>/w<W>_e<E>_g<epoch>_ep<K>` (spec 4.3); `match_log._worker_of` keeps parsing `w<N>_e`.

### `colosseum.net.security` (T1.3)

Identity dirs (`data_home()` = `$XDG_DATA_HOME` or `~/.local/share`; `cache_home()` = `$XDG_CACHE_HOME` or `~/.cache`):

```
<data_home>/colosseum/hub/                 hub identity (per machine, not per run)          0700
    ca.key   (ECDSA P-256 PEM, 0600)  ca.crt   (self-signed, CA:TRUE, SAN DNS:colosseum-hub; also the TLS server cert)
    secret   (token secret, 0600)     nodes.json   banned.json   ssh_nodes.json (T8.1)   running/<pid>.json (T2.5)
    rotated/<n>/                      (the previous CA after --rotate)
<data_home>/colosseum/node/<ca fingerprint>/   node identity                               0700
    node.key (0600)  node.crt  node.json  {"node_id", "number", "name", "secret" (0600 file mode), "hub_addresses": [...]}
<cache_home>/colosseum/bundles/<code hash>/   envs/<env key>/ (A32)   hub-bundles/   logs/<ca8>/<run_id>/, logs/<ca8>/join.log   (T6.x, T4.4, T7.2)
```

- Token: `cst1.<secret>.<fp>`; secret = 32 random bytes, base64url without padding (43 chars); fp = lowercase hex SHA-256 of the CA certificate's SubjectPublicKeyInfo (DER) (64 chars); regex `^cst1\.[A-Za-z0-9_-]{43}\.[0-9a-f]{64}$`. `ca8` = `fp[:8]`. The curl one-liner's `--pinnedpubkey 'sha256//<b64>'` is the standard base64 of the same digest.
- `nodes.json`: `{"version": 1, "next_number": int, "nodes": {node_id: {"number", "name", "machine_id", "tags", "secret_sha256", "cert_serial", "persistent": bool, "registered_run": run_id}}}` (node ids are 32 hex chars chosen by the hub at registration; numbers ≥ 1 never reused; 0 = `train`'s ephemeral local node, never stored (A9); names unique, `-2`, `-3` on collision). `banned.json`: `{"version": 1, "banned": [{"node_id", "machine_id", "name"}]}`.
- gRPC metadata keys: `x-colosseum-token` (token secret), `x-colosseum-node` (node id), `x-colosseum-node-secret`. Fixed server names `colosseum-hub` / `colosseum-node` (`grpc.ssl_target_name_override`).

```python
TOKEN_PREFIX = "cst1"; TOKEN_ENV = "COLOSSEUM_TOKEN"; HUB_SERVER_NAME = "colosseum-hub"; NODE_SERVER_NAME = "colosseum-node"
MD_TOKEN, MD_NODE_ID, MD_NODE_SECRET = "x-colosseum-token", "x-colosseum-node", "x-colosseum-node-secret"
class SecurityError(Exception)                           # refusals (exit 1 at the CLI, with a hint)
def data_home() -> Path; def cache_home() -> Path
def hub_identity_dir() -> Path; def node_identity_dir(ca_fingerprint: str, root: Path | None = None) -> Path
@dataclass(frozen=True)
class Token:
    secret: str
    ca_fingerprint: str
    def __str__(self) -> str
def parse_token(text: str) -> Token
def resolve_token(token: str | None, token_file: str | None, *, env: Mapping[str, str] = os.environ) -> Token  # flag > file > env
def spki_fingerprint(cert_pem: bytes) -> str
def curl_pin(fingerprint: str) -> str                    # "sha256//<base64>"
class HubIdentity:
    @classmethod
    def load_or_create(cls, path: Path | None = None) -> HubIdentity
    path: Path; fingerprint: str; ca_pem: bytes; token: Token; nodes: NodeBook
    def rotate(self) -> None                             # new CA + secret, every node revoked
    def issue_node_certificate(self, node_id: str, public_key_pem: bytes) -> bytes
    def server_credentials(self) -> grpc.ServerCredentials           # TLS, no client auth (R1)
class NodeBook:                                           # nodes.json + banned.json; thread-safe reads
    def register(self, *, machine_id: str, name: str, tags: Sequence[str], node_id: str | None, persistent: bool = True,
                 run_id: str = "") -> tuple[NodeRecord, str]   # (record, new secret)
    def add_ephemeral(self, name: str, machine_id: str) -> tuple[NodeRecord, str]   # number 0, in memory (A8, A9)
    def verify(self, node_id: str, secret: str) -> NodeRecord | None
    def ban(self, node_id: str) -> None; def is_banned(self, node_id: str | None, machine_id: str | None) -> bool
    def remove(self, node_id: str) -> None
class NodeIdentity:
    @classmethod
    def load_or_create(cls, ca_fingerprint: str, root: Path | None = None) -> NodeIdentity
    node_id: str | None; number: int | None; name: str | None; secret: str | None
    def public_key_pem(self) -> bytes; def save_registration(self, reply) -> None
def fetch_hub_ca(host: str, port: int, *, fingerprint: str, timeout: float) -> bytes   # unverified handshake, then pin check
def hub_channel_credentials(ca_pem: bytes) -> grpc.ChannelCredentials
def data_server_credentials(node: NodeIdentity, ca_pem: bytes) -> grpc.ServerCredentials   # mTLS, require_client_auth
def data_channel_credentials(node: NodeIdentity, ca_pem: bytes) -> grpc.ChannelCredentials
```

### `colosseum.net.grpc_util` (T1.3; blob parts T2.3)

```python
def channel_options(max_message_mb: int, *, server_name: str | None = None) -> list[tuple[str, Any]]   # sizes, keepalive 10 s
def server_options(max_message_mb: int) -> list[tuple[str, Any]]
class AuthInterceptor(grpc.ServerInterceptor):
    """R1. Open: Bootstrap/Info, Bootstrap/Script. Token secret: Bootstrap/Bundle, Hub/Register, Hub/Admin*.
    Node id + node secret (NodeBook.verify, not banned): every other Hub method. NodeData: the mTLS peer CN must be a
    known, not banned node id. Refusal = UNAUTHENTICATED / PERMISSION_DENIED with a hint in details.
    (A8: NodeData is refused on the hub port; data ports use DataPortInterceptor(allowed).)"""
    def __init__(self, token_secret: Callable[[], str], nodes: NodeBook) -> None
PART_BYTES = 1 << 20                                     # blob part size (T2.3)
def send_parts(data: bytes | memoryview | Path, *, meta_json: str = "", digest: str | None = None) -> Iterator[pb.BlobPart]   # A22
def receive_parts(parts: Iterable[pb.BlobPart], *, cap: int, into: Path | None = None) -> tuple[bytes | Path | None, str]   # digest checked; None = not_modified
```

### `colosseum.core.config` (T1.4)

```python
class DistributedBundleConfig(StrictModel):
    include: list[str] = []
    exclude: list[str] = []
    max_file_mb: float = 10.0                 # > 0
class DistributedConfig(StrictModel):        # global only (spec 9.1)
    port: int = 47100                         # 1..65535
    data_port: int = 47102                    # 1..65535, != port (ConfigError otherwise)
    insecure: bool = False
    heartbeat_sec: float = 1.0                # > 0
    node_lost_after_sec: float = 10.0         # > heartbeat_sec
    learner_relocate_after_sec: float = 120.0 # >= node_lost_after_sec
    hub_lost_timeout_sec: float = 1800.0      # > 0
    hub_lost_collect_sec: float = 300.0       # > 0
    shutdown_grace_sec: float = 20.0          # > 0
    connect_timeout_sec: float = 5.0          # > 0
    block_lease_sec: float = 600.0            # > 0
    relay: bool = True
    results_batch: int = 256                  # >= 1
    results_flush_sec: float = 0.25           # > 0
    node_result_buffer_mb: float = 256.0      # > 0
    max_message_mb: int = 64                  # >= 1 (was transport.grpc_max_message_mb)
    bundle: DistributedBundleConfig = DistributedBundleConfig()
    requirements: str | None = None           # a requirements file replacing the hub env freeze
class LearnerPlacementConfig(StrictModel):
    node: str | None = None                   # preferred node name
    tags: list[str] = []                      # preferred node tags (all must match)
    strict: bool = False                      # never relocate off the preference
class LearnerConfig(StrictModel):            # existing fields unchanged, then (per-agent override as today):
    max_policy_lag: int | None = None         # >= 0; None = off (spec 4.7)
    recovery_interval_sec: float = 300.0      # > 0
    placement: LearnerPlacementConfig = LearnerPlacementConfig()
class ColosseumConfig:                       # new section; `transport` is gone
    distributed: DistributedConfig
```

- `transport:` in the input YAML is translated by a `mode="before"` validator and never stored: `mode` / `grpc_port` → one WARNING, ignored; `grpc_max_message_mb` → `distributed.max_message_mb` (one WARNING); with `distributed.max_message_mb` also set → `ConfigError` (SP2/SP3 rule). `TransportConfig`, `TransportMode` and `ColosseumConfig.transport` are removed; `distributed.py` reads `config.distributed.max_message_mb` until T7.2.
- `learner.device` accepts `auto`, `cpu`, `cuda`, `cuda:N`, `mps` (validated; T4.1 uses it for placement).
- `--set distributed.x=...` works through the existing `apply_overrides`.

### `colosseum.node.weights` — shared-memory weight cache (T1.5)

Files in `shm_dir()` (`$COLOSSEUM_SHM_DIR`, else `/dev/shm` on Linux, else `<tempdir>/colosseum-shm` on macOS). Prefix `P = f"colosseum-{ca8}-{run_id[:8]}-n{node}"`:

| File | Content | Writer |
|---|---|---|
| `{P}-{agent}-v{N}` | weights frame (`encode_weights`, kind `latest`) | the agent's learner on this machine, or the node for a version fetched from another machine |
| `{P}-{agent}.latest` | JSON `{"segment", "version", "generation", "size", "digest", "teacher_active"}`, atomic replace | same single writer as the segments |
| `{P}-{agent}-{checkpoint_id}` | weights frame (kind `snapshot`) | node |
| `{P}-blob-{kind}-{blob_id}` | weights frame of a hub blob (`fixed` / `teacher` / `init` / `eval`; key `blob_key`, A18) | node |
| `{P}.lock` | JSON `{"pid", "proc_start"}` of the node process | node |

Segments are immutable; a reader `mmap`s read-only and builds `np.frombuffer` views; a reader that finds the pointer's segment gone re-reads the pointer (bounded retries). The writer of `latest` keeps the newest 2 versions and deletes older ones (POSIX: deleting after a reader's `mmap` is safe). Cleanup removes only files of prefixes whose lock names a dead process (`process_alive` semantics). Free space is checked before every write (`size × 2 + 64 MiB`), else `CacheFullError` (ERROR hint: `docker --shm-size`, `emptyDir: Memory`). `digest` = `hashlib.blake2b(frame, digest_size=16).hexdigest()`.

```python
SHM_DIR_ENV = "COLOSSEUM_SHM_DIR"
def shm_dir() -> Path
def cache_prefix(ca_fingerprint: str, run_id: str, node: int) -> str
@dataclass(frozen=True)
class CacheKey:
    kind: Literal["latest", "snapshot", "blob"]
    agent_id: str = ""
    ident: str = ""                     # checkpoint id (snapshot), f"{kind}-{blob_id}" (blob, A18), "" for latest
@dataclass(frozen=True)
class CachedWeights:
    key: CacheKey
    state_dict: dict[str, np.ndarray]    # read-only views over the mmap (load_state_dict copies)
    meta: WeightsMeta
    nbytes: int
class CacheFullError(RuntimeError)
class WeightCacheWriter:
    def __init__(self, root: Path, prefix: str) -> None
    def publish_latest(self, agent_id: str, frame: bytes, *, version: int, generation: int, teacher_active: bool) -> str  # segment name
    def put(self, key: CacheKey, frame: bytes) -> str
    def drop(self, key: CacheKey) -> None
class WeightCacheReader:
    def __init__(self, root: Path, prefix: str) -> None
    def latest_version(self, agent_id: str) -> tuple[int, int] | None          # (version, generation), pointer only
    def read_latest(self, agent_id: str, *, newer_than: int = -1) -> CachedWeights | None
    def get(self, key: CacheKey) -> CachedWeights | None
    def has(self, key: CacheKey) -> bool
def acquire_node_lock(root: Path, prefix: str) -> Path                         # writes {P}.lock
def cleanup_dead(root: Path) -> list[str]
```

### `colosseum.net.local` — same-machine connections and child IO (T2.2)

- Addresses: Linux uses abstract Unix sockets (`"\0" + name`, no files); macOS uses files in `$COLOSSEUM_RUNTIME_DIR` or `<tempdir>/cst-<uid>/` with paths ≤ 104 characters (ValueError otherwise). Names: `colosseum-{ca8}-{run8}-n{node}-node` (the node's listener for every child), `colosseum-{ca8}-{run8}-n{node}-learner-{agent}-i{incarnation}` (one per learner incarnation). Authkey: `os.urandom(32)` per node process, given to children in their spawn kwargs.
- Connections: each child opens one **control** connection to the node; a worker also opens one **route** connection to the node (frames for learners on other machines) and one **chunk** connection per local learner it collects for; the node opens one chunk connection to each local learner (frames from other machines). The first message on every connection is `ChildHello`.

```python
def listener(name: str, authkey: bytes) -> multiprocessing.connection.Listener
def connect(name: str, authkey: bytes, *, timeout: float) -> multiprocessing.connection.Connection
def address(ca8: str, run8: str, node: int, proc: str) -> str
# --- control messages (pickled dataclasses; R3) ---
@dataclass class ChildHello: role: Literal["worker", "learner", "evaluator", "node"]; name: str; incarnation: int; pid: int; channel: Literal["control", "route", "chunks"]
@dataclass class Stop: reason: str
@dataclass class ChildError: kind: Literal["contract", "player", "crash"]; message: str; context: str   # sent before the child exits
# node -> worker
@dataclass class CellLineup: env: int; lineup: Lineup; keys: dict[int, CacheKey]   # lineup.epoch set; keys per seat (R11)
@dataclass class WorkerLineups: cells: list[CellLineup]
@dataclass class WorkerEvict: keys: list[CacheKey]
@dataclass class LearnerRoute: agent_id: str; address: str | None; via_node: bool; generation: int   # None & not via_node: no collecting
@dataclass class WeightsReady: agents: list[str]                   # first-weights wait ends (agents without weights stay unplayed)
# worker -> node
@dataclass class WorkerResults: results: list[MatchResult]         # ResultBatcher batches (<= 256 or 0.1 s)
@dataclass class WorkerProgress: incarnation: int; env_steps: int; counters: dict[str, int]   # cumulative per incarnation, every 0.5 s
# node -> learner
@dataclass class LearnerEnvSteps: env_steps: int                   # the hub's global counter (LR progress)
# learner -> node
@dataclass class VersionPublished: agent_id: str; generation: int; version: int; segment: str; size: int; digest: str; teacher_active: bool
@dataclass class CheckpointHeader: kind: Literal["periodic", "final", "recovery"]; agent_id: str; generation: int; version: int; size: int
                                   # followed on the same connection by one send_bytes(encode_checkpoint(...))
@dataclass class LearnerMetrics: metrics: dict                      # one train-step dict (as today's metrics_queue items)
@dataclass class IntakeReport: depth: int; dropped: dict[str, int]  # cumulative, every 1 s
```

Worker side (used by `node.worker_main`, built over `LoopIO` — `RolloutLoop` keeps its loop, R4):

```python
class WorkerIO:
    """LoopIO over the control / route / chunk connections and a WeightCacheReader. send_chunk: frame (uncompressed)
    to the local learner's chunk connection (blocking: backpressure), or to the route connection (lz4 per Lz4Probe)
    for a remote learner, or dropped and counted (no_learner / disconnect); poll_weights: newer latest from the cache
    as a WeightPayload (generation set); poll_command: WorkerLineups / WorkerEvict / LearnerRoute merged into one
    WorkerCommand (snapshot weights read from the cache; R11 guarantees they are there); report_result: ResultBatcher
    over the control connection, each result stamped with node and seat generations; add_env_steps: cumulative counter
    sent as WorkerProgress. Keeps version -> generation of the loaded weights for chunk and seat generations."""
    def __init__(self, *, node: int, worker: int, incarnation: int, control: Connection, route: Connection,
                 cache: WeightCacheReader, routes: Mapping[str, LearnerRoute], authkey: bytes) -> None
    def loop_io(self) -> LoopIO
    def ready(self) -> bool                                          # WeightsReady received (first-weights predicate)
    def stopped(self) -> bool                                        # Stop received or the node connection closed
```

Learner side (R4: `learner_process` keeps its loop; queues become these two abstractions):

```python
class ChunkIntake(Protocol):
    def get(self, timeout: float) -> TrajectoryChunk                 # decoded and validated; raises queue.Empty on timeout (collect_batch)
    @property
    def depth(self) -> int                                           # frames waiting (system.queue_depths)
    @property
    def dropped(self) -> dict[str, int]                              # cumulative by reason (spec 4.7)
class LearnerOutputs(Protocol):
    def publish_weights(self, payload: WeightPayload) -> None
    def send_checkpoint(self, payload: dict, *, kind: Literal["periodic", "final", "recovery"], block: bool) -> bool
    def send_metrics(self, metrics: dict) -> None                    # never blocks
class ConnectionIntake:   # ChunkIntake: a reader thread waits on every chunk connection (Listener-accepted) and reads
                          # whole frames into a bounded deque of learner.queue_size frames; when full it stops reading
                          # (socket backpressure); get() decodes, validates (ChunkValidator), applies the generation
                          # rule and max_policy_lag (T3.3) and counts drops; an EOF / partial frame closes only that
                          # connection (counted "disconnect"); the control connection's LearnerEnvSteps / Stop are read
                          # by the same thread into env_steps / stop_flag.
    def __init__(self, listener: Listener, control: Connection, validator: ChunkValidator, *, agent_id: str,
                 generation: int, version_floor: int, capacity: int, cap_bytes: int) -> None
    env_steps: int; stop_flag: threading.Event
class NodeOutputs:        # LearnerOutputs: weights -> WeightCacheWriter.publish_latest + VersionPublished;
                          # checkpoints -> CheckpointHeader + frame on the control connection; metrics -> LearnerMetrics
    def __init__(self, control: Connection, cache: WeightCacheWriter, *, agent_id: str, generation: int) -> None
class QueueIntake: ...; class QueueOutputs: ...   # the SP4 queue path (launcher until T2.5, distributed.py until T7.2, tests)
```

### Changes to existing loops (T2.2)

```python
# colosseum.learner.learner
def learner_process(*, agent_id: str, algorithm_factory: Callable[[], BaseAlgorithm], intake: ChunkIntake,
                    outputs: LearnerOutputs, config: LearnerConfig, stop_event: Any,       # .is_set(), .wait(t)
                    checkpoint_interval: int = 0, resume_state: dict | None = None,
                    progress_counter: Any | None = None,   # .value: the global env-step count (SharedCounter or ConnectionIntake)
                    total_timesteps: int = 0, weight_sync_interval: float = 5.0,
                    version_floor: int = 0,                # R9: policy_version = max(restored, version_floor) before the first publish
                    recovery_interval_sec: float | None = None,   # T4.2
                    ) -> None
def collect_batch(q: Any, batch_size: int, stop_event: Any, poll_interval: float = 0.5,
                  agent_id: str | None = None) -> list[TrajectoryChunk] | None   # items may be TrajectoryChunk or payload dicts
def set_policy_version(algorithm: BaseAlgorithm, version: int) -> None
# colosseum.core.types
@dataclass class WeightPayload: ...; generation: int = 0
# colosseum.worker.match_runner.MatchRunner / colosseum.worker.rollout_loop.RolloutLoop: new keyword
cell_seed_fn: Callable[[int, int], int | None] | None = None   # (env, epoch) -> cell seed; used only for lineups with an epoch
# RolloutLoop only: uid_prefix: str | None = None   # None = SP4's f"{run_id}/train/"; node workers pass f"{run_id}/train/n{node}/"
# colosseum.utils.seeding
def cell_seed(seed: int | None, epoch: int, node: int, worker: int, env: int) -> int | None   # spec 7.6 (sha256 like derive_seed)
def worker_seed(seed: int | None, node: int, worker: int, incarnation: int) -> int | None      # process RNG seed
```

### `colosseum.hub.state` — pure hub state (T2.1)

```python
@dataclass(frozen=True)
class CellKey:
    node: int
    worker: int
    env: int
@dataclass
class NodeEntry:                       # one registered node as the hub sees it
    node_id: str; number: int; name: str; session_id: str | None; state: Literal["online", "lost", "removed"]
    hardware: dict; slots: dict; tags: tuple[str, ...]; data_addresses: tuple[str, ...]; data_path: Literal["direct", "relay", "local"]
    last_heartbeat: float              # hub monotonic
class NodeRegistry:
    def connect(self, node_id, number, name, session_id, hello, now) -> Literal["new", "resumed", "returned"]
    def heartbeat(self, node_id, session_id, now) -> bool; def lost(self, now, after_sec) -> list[NodeEntry]
    def online(self) -> list[NodeEntry]; def get(self, number) -> NodeEntry | None
class EpochAllocator:                  # R7
    def __init__(self, reserved_until: int, on_reserve: Callable[[int], None]) -> None   # on_reserve saves hub_state.json (fsync) first
    def next(self) -> int
class OwnerRotation:                   # spec 4.2 deficit rule
    def __init__(self, agents: Sequence[str]) -> None                       # config order
    def assign(self, cells: Sequence[CellKey], *, collecting: Sequence[str],
               owners: Mapping[CellKey, str], previous: Mapping[CellKey, str]) -> dict[CellKey, str]
       # cells: the requested ones, in (node, worker, env) order; owners: current owners of every other live collecting cell.
       # deficit = (1 / len(collecting)) * live collecting cells - owned; tie -> the first tied agent after the cell's
       # previous owner in config order (no previous owner: the first tied agent). One node, simultaneous expiry:
       # reproduces trainable[(g + round) % n] (T0.1 sequence test).
@dataclass
class CellEntry: key: CellKey; epoch: int; incarnation: int; owner: str | None; previous: str | None; granted_at: float; ttl_sec: float
class CellTable:
    def request(self, node: int, cells: Sequence[tuple[CellKey, int, int]], now: float) -> list[CellEntry]  # (key, epoch, incarnation)
       # same incarnation + an epoch this hub incarnation granted: the epoch stays; otherwise a new epoch
    def drop_node(self, node: int) -> list[CellEntry]; def live(self) -> list[CellEntry]
class BudgetCounter:                   # spec 4.4
    def __init__(self, base: int = 0, sessions: Mapping[str, int] | None = None) -> None   # key f"{node_id}/{session_id}"
    def report(self, node_id: str, session_id: str, cumulative: int) -> int               # applied increment (>= 0)
    @property
    def value(self) -> int
    def state(self) -> dict
@dataclass(frozen=True)
class VersionInfo:                     # defined in colosseum.net.messages (T1.2), re-exported here (A7)
    agent_id: str; generation: int; version: int; size: int; digest: str; node: int   # node -1: parked (A27)
    addresses: tuple[str, ...] = (); relay_only: bool = False; teacher_active: bool = False
class VersionRegistry:                 # spec 4.5, 7.2
    def new_generation(self, agent_id: str) -> int          # increments; records cap(previous) = max accepted version
    def generation(self, agent_id: str) -> int              # current; 0 = none yet
    def announce(self, info: VersionInfo) -> Literal["accepted", "stale_generation", "not_newer", "invalid"]
    def latest(self, agent_id: str) -> VersionInfo | None
    def version_floor(self, agent_id: str) -> int           # max accepted + 1 (0 without any)
    def is_stale(self, agent_id: str, generation: int | None, version: int | None) -> bool   # cap rule (Q2)
    def state(self) -> dict; @classmethod def from_state(cls, d: dict) -> VersionRegistry
class ResultIntake:                    # spec 4.3, R8
    def __init__(self, *, journal: bool) -> None
    def check(self, result: MatchResult, *, spec: GameSpec, agents: Collection[str], versions: VersionRegistry,
              node: int | None = None, uid_prefix: str | None = None) -> str | None
       # None = valid; "invalid" (layout / seats / roles / unknown agent / non-finite numbers / ...), "stale_generation"
    def begin(self, node_id: str, session_id: str, seq: int) -> bool       # False: seq acked or in flight (skip)
    def committed(self, node_id: str, session_id: str, seq: int) -> None   # -> ack
    def failed(self, node_id: str, session_id: str, seq: int) -> None      # not acked: the node resends
@dataclass
class HubStateData:                    # hub_state.json
    version: int; run_id: str; env_steps: int; sessions: dict[str, int]; epoch_reserved_until: int
    versions: dict; counters: dict[str, Any]; nodes_lost: int
def save_hub_state(path: Path, data: HubStateData, *, fsync: bool) -> None; def load_hub_state(path: Path) -> HubStateData | None
HUB_STATE_FILE = "hub_state.json"; HUB_STATE_VERSION = 1
```

`hub_state.json`: the schema of A3 (`counters` = `{"book", "intake_book"}`, the `CounterBook` states of A2; `learners` from T4.1) — written atomically on every metrics tick (`fsync` only for epoch reservations) and at the end (T2.5); loaded by `--resume` (T4.3).

### `colosseum.hub.server` — gRPC servicers and inbox events (T2.3; relay T3.2; bootstrap T6.2; evaluation T5.1)

```python
def build_server(*, inbox: queue.SimpleQueue, identity: HubIdentity, insecure: bool, listen: Sequence[str],
                 loopback_plain: bool, blobs: BlobStore, max_message_mb: int,
                 bootstrap: BootstrapService | None = None, relay: RelayHub | None = None, latest: LatestRelay | None = None,
                 policy: Callable[[], RegistrationPolicy] | None = None) -> tuple[grpc.Server, dict[str, int]]   # A22
class SessionHandle:                   # one live session stream
    node_id: str; session_id: str
    def send(self, msg: pb.HubMessage) -> None          # thread-safe; per-session outbound queue (bounded; overflow closes the session)
    def close(self, reason: str) -> None
class BlobStore:                       # read-only file serving + lazy frame conversion (servicer threads)
    def __init__(self, run_dir: RunDir) -> None
    def frame_path(self, kind: str, agent_id: str, ident: str) -> Path | None   # runs/<name>/blobs/<kind>-<agent>-<ident>.frame
    def register_fixed(self, blob_id: str, frame: bytes, *, kind: str, agent_id: str) -> None   # main loop at start; more methods: A22
# inbox events (dataclasses; servicer threads -> main loop)
@dataclass class SessionOpened: handle: SessionHandle; hello: pb.Hello
@dataclass class SessionClosed: node_id: str; session_id: str; reason: str
@dataclass class NodeMessageIn: node_id: str; session_id: str; message: pb.NodeMessage   # every oneof except hello
@dataclass class RegisterIn: request: pb.RegisterRequest; peer: str; reply: concurrent.futures.Future
@dataclass class UploadIn: header: pb.UploadHeader; path: Path; node_id: str; reply: concurrent.futures.Future   # spooled in runs/<name>/.incoming/
@dataclass class AdminIn: command: str; args: dict; reply: concurrent.futures.Future | queue.SimpleQueue
# RelayEvent: not used (A22: relay counters come from RelayHub.stats())
```

Result path: the servicer converts a `ResultBatch` with `results_from_batch` in its thread and posts `NodeMessageIn` with the converted list attached (`message`, `results: list[MatchResult]`); the main loop applies `ResultIntake.check`, stamps `env_steps` (hub counter at intake), and submits (R8 below).

### `colosseum.hub.hub` — the hub process (T2.5; R5)

```python
@dataclass(frozen=True)
class LocalNodeOptions:                # R12: train's local node (a child of the hub)
    workers: int                       # rollout.num_workers
    evaluators: int                    # 0 until T5.1, then 1 (a slot run on demand, A10); from_config(config, run_dir)
    log_dir: Path                      # runs/<name>/logs
    data_port: bool                    # True only with train --listen (-> NodeOptions.data_port = -1)
@dataclass(frozen=True)
class HubOptions:
    mode: Literal["train", "hub", "idle"] = "train"
    listen: str | None = None          # train: None = loopback only; hub: None = all interfaces
    port: int | None = None            # None: distributed.port (train without --listen: an ephemeral loopback port)
    insecure: bool = False
    resume: bool = False               # T4.3: in-place resume of run_dir
    local_node: LocalNodeOptions | None = None
    identity_dir: Path | None = None   # default security.hub_identity_dir()
    nodes_file: Path | None = None     # T8.1
    dashboard_port: int | None = None  # T7.1
    bootstrap: bool = True             # T6.3: False = no BootstrapService (cluster_kit; A10)
class Hub:
    def __init__(self, config: ColosseumConfig, run_dir: RunDir, *, options: HubOptions, validated: bool = False) -> None
    @property
    def env_steps_done(self) -> int
    def request_stop(self) -> None     # thread-safe (bench, tests); exit code 0
    def launch(self) -> int            # 0 / 1 / 130 / 143
def run_training(config_path: str | None, overrides: dict | None = None, *, resume: str | None = None,
                 listen: str | None = None, nodes_file: str | None = None) -> int   # keywords added by their tasks (A10)
def run_hub(config_path: str | None, overrides: dict | None = None, *, resume: str | None = None, idle: bool = False,
            listen: str | None = None, port: int | None = None, insecure: bool = False, nodes_file: str | None = None,
            dashboard: int | None = None) -> int
# moved from launcher.py unchanged: RunSetup, setup_run, validate_run_config, warn_static_ownership_skew
# colosseum.node.worker_main: create_env (was launcher._create_env)
```

- R5: `Hub` keeps `Launcher`'s method names where the semantics stay: `_build_coordinator`, `_open_metrics`, `_open_match_log`, `_open_live_ratings`, `_wire_retention`, `_reapply_retention`, `_live_snapshot`, `_stored_snapshot_versions`, `_finish_live_ratings`, `_on_snapshot_evicted`, `_match_log_stats`, `_stop_match_log`, `_finish_metrics`, `_resolve_resume`, `_resolve_init`, `_load_league_state`, `_save_league_state`, `_import_snapshot_pool`, `_save_checkpoint`, `_pump_pulse`, `_write_pulse_state`, `_finish_pulse`, `_shutdown`, `_exit_code_after_shutdown`, `_signal_exit_code`. Queue drains become handlers: `_on_result_batch`, `_on_metrics_batch`, `_on_upload`, `_on_version`, `_on_event`, `_on_lineup_request`, `_on_session_opened`, `_on_session_closed`, `_on_register`, `_on_admin`. `_refresh_worker_matches` / `_resolve_lineups` are gone (pull, R11).
- **Main loop:** `inbox.get(timeout=INBOX_POLL_SEC)` and drain; then on monotonic deadlines: node loss (`node_lost_after_sec`), lineup TTL bookkeeping, commit intake (`MatchLogWriter.take_commits`), retention after a refit, pulse feeding, metrics tick (+ `hub_state.json`, `league_state.json`, `nodes.json` from T7.1), budget (`env_steps >= total_timesteps` → stop), signals.
- **Lineups:** a `LineupRequest` is answered in the same loop pass: `CellTable.request` → `OwnerRotation.assign` over agents with a running learner (none: only non-collecting lineups, and none at all while no agent can collect and no fixed-only work exists; spec 4.2) → `Coordinator.lineup_for(owner)` → `LineupGrants` with `epoch` and `ttl_sec = rollout.match_refresh_interval_sec` (0 = whole run). `match_refresh_interval_sec: 0` grants each cell once.
- **Accounting (R8):** with the journal: `MatchLogWriter.submit(valid_results, token=(node_id, session_id, seq))`; a submit the writer cannot take whole returns its size, the batch is neither accounted nor acked (resent); each `Commit` from `take_commits()` is accounted in commit order — `Coordinator.report_match_result` and `MetricsHub.on_match_result` for every newly journaled result (duplicates skipped) — then the batch is acked; `league_state.json`'s `journal_row` = the last accounted commit's highest row id. A writer that is dead (journal unusable) switches to the journal-off rule with one ERROR, counted in `dropped_match_results_by_reason.writer`. Without the journal: accounting at intake, ack at once, dedup by (node, session, seq) only. Results counted in counts only (disk guard) are accounted at commit but cannot be replayed (documented).
- **Ports:** `train` without `--listen`: one plain loopback port (ephemeral); `train --listen [ADDR]`: TLS on `ADDR:distributed.port` plus the plain loopback port for the local node; `hub`: TLS on all interfaces (`--listen` narrows) at `--port` / `distributed.port`; `--insecure`: plain everywhere. Every mode requires the token secret.
- **Local node (R12):** `NodeAgent` code started as a spawned child (`node.agent.local_node_target`) with in-process options: the hub's `sys.path` and cwd, the hub identity's token, the plain loopback address, `LocalNodeOptions`; logs in `runs/<name>/logs/` (`main.log` = hub, `node.log`, `worker-<i>.log`, `learner-<agent>.log`, `evaluator-<i>.log`). Every trainable agent's learner is assigned to it (T2.5 assigns directly; T4.1 through `placement.place`).

### `colosseum.hub.registry` (T2.5)

```python
HUB_JSON = "hub.json"
@dataclass(frozen=True)
class HubInfo:
    run_id: str; run_dir: str; pid: int; proc_start: int | None; addresses: tuple[str, ...]   # "host:port"
    loopback: str | None; tls: bool; ca_fingerprint: str; identity_dir: str; mode: str
def write_hub_json(run_dir: RunDir, info: HubInfo) -> Path      # runs/<name>/hub.json (atomic; kept after the stop)
def read_hub_json(run_dir: str | Path) -> HubInfo | None
def register_running(identity_dir: Path, info: HubInfo) -> Path  # <identity>/running/<pid>.json
def unregister_running(identity_dir: Path, pid: int) -> None
def running_hubs(identity_dir: Path) -> list[HubInfo]           # live pids only; stale entries removed
```

### `colosseum.hub.admin` (T2.5; commands added by T5.2, T6.3, T8.x)

```python
@dataclass(frozen=True)
class HubTarget: address: str; tls: bool; ca_pem: bytes | None; token: Token
def resolve_hub(*, hub: str | None, run: str | None, token: str | None, token_file: str | None, insecure: bool = False) -> HubTarget
   # spec 3.3: --hub ADDR (+ token) > --run RUN (hub.json) > the single running hub of this machine (registry);
   # several running: SecurityError listing them; on the hub machine the local identity supplies the token
def call(target: HubTarget, command: str, args: Mapping[str, Any] | None = None, *, timeout: float = 30.0) -> dict   # {code, text, data} (A5)
def stream(target: HubTarget, command: str, args: Mapping[str, Any] | None = None) -> Iterator[dict]
ADMIN_COMMANDS = ("hub.status", "nodes.list", "nodes.remove", "tournament.submit", "tournament.status", "tournament.cancel",
                  "nodes.add", "nodes.update_code", "agents.add", "agents.freeze", "agents.remove", "agents.list")
```

### `colosseum.hub.placement` (T4.1)

```python
@dataclass(frozen=True)
class LearnerSlot: node: int; device: str                 # "cpu" | "cuda:N" | "mps"
@dataclass(frozen=True)
class NodeCapacity: node: int; name: str; tags: tuple[str, ...]; gpus: tuple[int, ...]; cpu_learners: int; unlimited: bool; mps: bool = False
@dataclass(frozen=True)
class PlacementPlan: assigned: dict[str, LearnerSlot]; pending: list[str]; reasons: dict[str, str]
def place(agents: Mapping[str, LearnerConfig], nodes: Sequence[NodeCapacity], *, current: Mapping[str, LearnerSlot],
          relocate: Collection[str] = ()) -> PlacementPlan
   # device auto -> a GPU slot if any node has a GPU, else CPU; cuda[:N] -> a GPU slot (index N); cpu -> a CPU slot;
   # placement.node / tags preferred, strict forbids others; a learner never moves while its node lives (only `relocate`)
def describe_plan(config: ColosseumConfig) -> list[str]   # validate's placement lines (T7.2)
```

### `colosseum.hub.evaluation` (T5.1, T5.2)

```python
@dataclass(frozen=True)
class PlayerRef: name: str; agent_id: str; network_id: str; roles: tuple[str, ...]; kind: Literal["neural", "scripted"]
                 blob: CacheKey | None = None; networks: dict | None = None; bot: BotSpec | None = None
@dataclass(frozen=True)
class BlockTask: block_id: str; source: Literal["pulse", "tournament"]; owner: str; lineups: list[Lineup]
                 players: list[PlayerRef]; priority: int; deterministic: bool   # priority 0 tournament, 1 pulse
@dataclass(frozen=True)
class Lease: lease_id: str; node: int; block_ids: tuple[str, ...]; deadline: float      # hub monotonic
class BlockQueue:                      # pure; hub clock
    def submit(self, tasks: Sequence[BlockTask]) -> None
    def lease(self, node: int, slots: int, now: float, lease_sec: float) -> list[tuple[Lease, list[BlockTask]]]
    def complete(self, lease_id: str, block_id: str, results: list[MatchResult], now: float) -> Literal["accepted", "late", "cancelled", "unknown"]
    def expire(self, now: float) -> list[str]; def drop_node(self, node: int) -> list[str]
    def cancel(self, owner: str) -> tuple[list[str], dict[int, list[str]]]   # (block ids, {node: [lease ids]}); more: A31
class HubPulse: ...                    # PulseFeeder snapshots -> blocks; T5.1 (replaces the pulse child, R6)
class HubTournament: ...               # a tournament driven in its own thread; play() -> inbox + future; T5.2
```

### `colosseum.node` (T2.3–T2.4; later tasks extend)

```python
# node.session (T2.3)
class HubSession:
    """Session client: connect (TLS / insecure / loopback), Hello -> Welcome, a reader thread posting HubMessages into
    the node inbox, send() thread-safe, reconnect with backoff RECONNECT_MIN_SEC -> RECONNECT_MAX_SEC; after a reconnect
    the caller resends unacked result batches."""
    def __init__(self, address: str, *, credentials: grpc.ChannelCredentials | None, node_id: str, node_secret: str,
                 hello: Callable[[], pb.Hello], inbox: Callable[[object], None], max_message_mb: int) -> None
    def start(self) -> None; def send(self, msg: pb.NodeMessage) -> bool; def close(self) -> None
    @property
    def connected(self) -> bool
# node.agent (T2.4)
@dataclass(frozen=True)
class NodeOptions:
    hub_address: str; token: Token; insecure: bool = False; loopback_plain: bool = False
    workers: int | None = None; evaluators: int | None = None; cpu_learners: int | None = None
    no_learners: bool = False; gpus: tuple[int, ...] | None = None; name: str | None = None; tags: tuple[str, ...] = ()
    once: bool = False; identity_root: Path | None = None; log_dir: Path | None = None
    local: bool = False                # train's local node (R12): unlimited learner slots, no Bootstrap, hub's sys.path
    data_port: int | None = None       # None: no data port; 0 ephemeral; -1 distributed.data_port (A23)
    # added: credentials, shutdown_grace_sec (T2.4), data_addresses (T3.1), code_hash, bundle_dir (T6.3): A23
class NodeAgent:
    def __init__(self, options: NodeOptions) -> None
    def run(self) -> int               # exit codes of spec 7.5 (join)
    def request_stop(self) -> None
def local_node_target(**kwargs) -> None                 # spawn target used by Hub (R12)
def main(argv: Sequence[str] | None = None) -> int     # `python -m colosseum.node.agent` (exec'd by the loader, T6.2)
# node.procs (T2.4)
@dataclass(frozen=True)
class ChildSpec: name: str; role: Literal["worker", "learner", "evaluator"]; target: Callable; kwargs: dict; daemon: bool
class ChildSupervisor:                 # spawn, per-name incarnations, restart limits (WORKER_RESTARTS / LEARNER_RESTARTS),
                                       # terminate/kill with a bound; children run under one try/finally (no leaks)
    def start(self, spec: ChildSpec) -> int; def exited(self) -> list[tuple[str, int | None]]
    def may_restart(self, name: str, now: float) -> bool; def stop_all(self, grace: float) -> list[str]
def watch_parent() -> None             # T4.4: PR_SET_PDEATHSIG on Linux, a watchdog thread on macOS
# node.hardware (T2.4)
@dataclass(frozen=True)
class Hardware: logical_cores: int; physical_cores: int; memory_bytes: int; gpus: tuple[dict, ...]; os: str
                platform: str; python: str; torch_version: str; torch_build: str
def probe() -> Hardware                # cgroup quota and affinity aware
def default_slots(hw: Hardware) -> dict   # workers = physical - 1 (>= 1); evaluators / cpu_learners = 1 if physical >= 4 else 0; one per GPU
def machine_id() -> str                # /etc/machine-id, IOPlatformUUID on macOS, else hostname hash
# node.logs (T2.4; tails T4.4)
def node_log_dir(ca8: str, run_id: str) -> Path   # <cache_home>/colosseum/logs/<ca8>/<run_id>/
def log_tail(path: Path, limit: int = LOG_TAIL_BYTES) -> bytes
# node.worker_main / node.learner_main / node.evaluator_main (T2.4 / T5.1)
def worker_target(**kwargs) -> None; def learner_target(**kwargs) -> None; def evaluator_target(**kwargs) -> None
# node.router (T3.2)
class ChunkRouter: ...                 # frames for remote learners: direct data-port stream per target node, else relay
# DataServer lives in node.dataport (T3.1; Chunks T3.2): A25
```

Node rules that cross tasks:
- **Lineup after cache (R11, spec 4.2):** a grant is passed to its worker only when every key it names (`latest` of each latest seat's agent, the snapshot of each `ckpt_v<N>` seat, the blob of each frozen seat) is in the cache; until then the env keeps its lineup; a failed fetch keeps the old lineup, counts `lineup_fetch_failed` and re-requests at the next pass.
- **Fixed players:** the node fetches every frozen agent's and teacher's blob listed in `Welcome.fixed` before it starts workers; workers get `FixedPlayers` with views from the cache.
- **Results:** batches of ≤ `distributed.results_batch` results or every `results_flush_sec`, seq per session from 1; unacked batches kept up to `node_result_buffer_mb` (estimated encoded size), newest dropped on overflow (counted, WARNING once); resent in seq order after every reconnect.
- **Env steps:** the node sums its workers' cumulative `WorkerProgress.env_steps` over incarnations into one cumulative per session, sent in every `Heartbeat`; it forwards `Announce.env_steps` to its learners (`LearnerEnvSteps`).
- **Until T3.2** the hub grants collecting seats only for agents whose learner runs on the requesting node; **until T3.1** latest weights exist only on the learner's machine.

### Same-machine data flow and where per-chunk cost lands (criterion 2)

| Flow | Path | Per-item cost on the hot path |
|---|---|---|
| chunk (local learner) | worker `encode_chunk` (no lz4) → `send_bytes` on its own connection → learner reader thread `recv_bytes` into the bounded deque → learner main `decode_chunk` + `ChunkValidator` | replaces pickle of the payload dict + `mp.Queue` feeder; decode is the criterion-5 path (≥ 10× faster than npz) |
| weights | learner `encode_weights` + one shm write per push (every `weight_push_interval` steps) → workers read the pointer every `weight_sync_interval_sec` | replaces one pickle per worker mailbox per push (and the stale unpickle) |
| results | worker `ResultBatcher` (≤ 256 / 0.1 s) pickled to the node → node batch (≤ 256 / 0.25 s) → protobuf over loopback gRPC → servicer thread converts (≈ 3.4 µs/result) → main loop intake → writer → commit accounting | ≈ 4 000 results/s at 4 workers: < 0.1 core in the hub |
| lineups | one `LineupRequest` per worker per refresh (30 s) over loopback gRPC | negligible |
| env steps | `WorkerProgress` every 0.5 s → `Heartbeat` every `heartbeat_sec` | the bench sets `distributed.heartbeat_sec: 0.25` so counter quantization stays < 1 % of its 60 s window (T2.5) |
| queue depth | learner `IntakeReport.depth` (frames waiting) → `system.queue_depths` | keeps the bench's learner-bound classifier |

### `colosseum.arena.matchlog` changes (T2.5; T4.3)

```python
@dataclass class AppendStats: ...; rows: list[int | None] = field(default_factory=list)   # matches.id per result (None: duplicate / counts only)
@dataclass(frozen=True)
class Commit:
    token: Any                       # what submit() was given
    accepted: list[MatchResult]      # newly journaled results, in commit order
    duplicates: int
    last_row: int | None             # highest matches.id of this commit
    failed: bool                     # write error: nothing of the token is accounted or acked
class MatchLogWriter:
    def submit(self, results: Sequence[MatchResult], *, token: Any = None) -> int   # with a token: all or nothing
    def take_commits(self) -> list[Commit]                                         # main loop; tokens whose last result is written
SCHEMA_VERSION = 2                    # T4.3: matches.node INTEGER (NULL for SP4 rows); v1 files migrated in place on open and on import
```

`league_state.json` gains `"journal_row": int | None` (T2.5 writes, T4.3 replays `train` rows after it into the coordinator on `--resume`); `Coordinator.save_league_state(path, *, env_steps, journal_row=None)`, `load_league_state(path) -> bool` unchanged, `Coordinator.league_state_journal_row() -> int | None` (T4.3). `Coordinator.lineup_for(owner: str) -> Lineup` (matchmaker + `check_lineup`; T2.5); `generate_lineups` stays for tests.

### Metrics (T2.5, T3.3, T4.1, T7.1)

- Worker stats are keyed by `(node, worker, incarnation)` (T2.5): the dict a node forwards carries `node`, `worker_id`, `incarnation`; `SystemStats` keeps the max of each cumulative counter per key and sums the keys, so `dropped_reward_episodes` / `dropped_match_results` stay monotonic across joins, restarts and in-place resume (saved in `hub_state.json` counters).
- `system` (T7.1; old keys kept, `REQUIRED_KEYS["system"]` extended with every key here): `nodes` (online now), `nodes_lost` (cumulative), `by_node` (`{"<number>": {"name", "workers", "env_steps_per_sec", "results_per_sec", "load", "last_heartbeat_age_sec", "data_path"}}`), `relay_bytes` (cumulative), `dropped_chunks` (`{"disconnect", "no_learner", "stale", "stale_generation", "invalid", "overflow"}` cumulative), `dropped_match_results_by_reason` (`{"node_buffer", "invalid", "stale_generation", "writer"}`; the int `dropped_match_results` stays their sum + workers' drops), `lineup_fetch_failed`, `worker_restarts`, `learner_restarts`, `pending_agents` (list of ids).
- `train` per agent: `policy_lag_p50`, `policy_lag_p95`, `policy_lag_max` (existing), `dropped_stale_chunks` (T3.3); `learner_node` (node number), `generation` (T4.1).
- `runs/<name>/nodes.json` (T7.1, each metrics tick): `{"env_steps", "nodes": [{"number", "name", "node_id", "state", "hardware", "roles": {"workers", "evaluators", "learners": [agent]}, "env_steps_per_sec", "results_per_sec", "load", "last_heartbeat_age_sec", "data_path", "tags", "code_hash"}]}`.
- Console: one line per agent (as today) + one nodes line `nodes: <online> online, <lost> lost, <env steps/s> env steps/s` (T7.1).
- Report / dashboard: section `nodes` added to `arena.report.data.SECTIONS` (T7.1); the training section shows policy lag.

### Run-dir files added by SP5

| File | Content | Task |
|---|---|---|
| `hub.json` | `HubInfo` | T2.5 |
| `hub_state.json` | counters, sessions, epoch reservation, versions, generations | T2.1 / T2.5 |
| `blobs/*.frame` | lazily converted snapshot / fixed / init / teacher weights frames | T2.3 |
| `.incoming/` | spooled uploads (removed after handling) | T2.3 |
| `checkpoints/.evicted/<agent>/ckpt_v<N>/` | hard links of evicted snapshots still referenced (≤ 2 × refresh) | T3.1 |
| `recovery/<agent>/recovery.frame` + `meta.json` | the newest recovery point (`{"generation", "version", "env_steps"}`), overwritten | T4.2 |
| `config.resolved.<k>.yaml` | the previous resolved config of an in-place resume | T4.3 |
| `agents_state.json` | agent set with status and start points (read by T4.3 when present) | T8.2 |
| `nodes.json` | node registry snapshot | T7.1 |
| `logs/nodes/<node>/<proc>.log.tail` | last 64 KB of a dead remote process's log | T4.4 |

`run_state.json` keeps its SP4 shape (the hub's pid).

### CLI (spec 9.2) and exit codes

| Command | Task |
|---|---|
| `colosseum train -c CFG [--set k=v ...] [--listen [ADDR]]` (hub + local node) | T2.5 |
| `colosseum train --resume RUN [-c CFG] [--set ...]` (`-c` optional with `--resume`) | T4.3 |
| `colosseum train ... --nodes FILE` | T8.1 |
| `colosseum hub -c CFG [--listen ADDR] [--port P] [--insecure] [--set ...]` (`hub` is a click group invoked without a subcommand) | T2.5 |
| `colosseum hub --resume RUN [-c CFG]` | T4.3 |
| `colosseum hub --idle -c CFG` | T5.2 |
| `colosseum hub --dashboard [PORT]` (default 8765, 127.0.0.1) | T7.1 |
| `colosseum hub --nodes FILE` | T8.1 |
| `colosseum hub token [--rotate]` | T6.3 |
| `colosseum join ADDR (--token T \| --token-file F) [--workers N] [--evaluators N] [--cpu-learners N] [--no-learners] [--gpus LIST] [--name NAME] [--tags a,b] [--once] [--insecure]` (token also from `COLOSSEUM_TOKEN`) | T6.3 |
| `colosseum nodes list`, `nodes remove NAME [--ban]` (+ `--hub ADDR` / `--run RUN` / `--token` / `--token-file`) | T6.3 |
| `colosseum nodes add USER@HOST [--workers N] [--evaluators N] [--name NAME] [--tags a,b] [--ssh-opt OPT ...]` | T8.1 |
| `colosseum nodes update-code [--restart-learners]` | T8.3 |
| `colosseum agents add --id X -c FRAGMENT`, `agents freeze X`, `agents remove X`, `agents list` | T8.2 |
| `colosseum tournament ... --hub ADDR\|RUN_DIR [--token T] [--cancel NAME]` | T5.2 |
| `colosseum serve-weight-store` / `run-learner` / `run-workers` → exit 2 + hint "use colosseum hub / join / nodes" | T7.2 |
| `colosseum validate` prints the placement plan and a `distributed:` line (port, TLS mode, relay, timeouts) | T7.2 |

Exit codes (spec 7.5): `hub` / `train` 0 budget reached (or `request_stop`) / 1 config error, contract error, a `failed` agent at the end, fatal hub error / 2 bad arguments / 130 SIGINT / 143 SIGTERM. `join` 0 normal stop (`--once` after the run, `nodes remove`) / 1 version mismatch it cannot update from, access refused, fatal / 2 / 130 / 143. `nodes …`, `agents …`, `hub token` 0 / 1 (hub error or refusal) / 2. `tournament --hub` 0 finished (also `confirmed: false`) / 130 detached / 1 cancelled or failed / 2.

### Bootstrap (T6.1, T6.2)

```python
# bootstrap.bundle (T6.1)
@dataclass(frozen=True)
class Bundle: path: Path; code_hash: str; files: tuple[str, ...]; size: int; python_path: tuple = (); skipped: tuple = ()   # A32
def build_bundle(project_dir: Path, *, include: Sequence[str], exclude: Sequence[str], max_file_mb: float,
                 out_dir: Path) -> Bundle      # tar of colosseum/ (package dir), the project dir (minus runs/, .venv/,
                                               # .git/, __pycache__/, .superpowers/, files > max_file_mb, exclude) and
                                               # editable packages outside the project; sorted names, zeroed metadata;
                                               # code_hash = sha256 hex of the archive bytes
def unpack_bundle(archive: Path, code_hash: str, *, cache_root: Path) -> Path   # <cache>/colosseum/bundles/<hash>/ (verified)
# bootstrap.requirements (T6.1)
EXCLUDED = ("torch", "nvidia-*", "triton", "colosseum", "cuda-*", "pytorch-triton*")
def node_requirements(*, override: Path | None) -> tuple[list[str], str]   # (pinned list, requirements hash)
# bootstrap.torch_select (T6.1)
def torch_spec(hub_torch: str, hub_build: str, *, platform: NodePlatform, gpus: Sequence[NvidiaGpu]) -> TorchChoice   # A32
# bootstrap.installer / loader (T6.2)
def installer_script(*, insecure: bool, python: str, grpcio: str, uv_version: str = UV_VERSION) -> bytes   # address: COLOSSEUM_HUB (A32)
   # whole body inside _colosseum_install() { ... }; the last line `_colosseum_install "$@" # colosseum-installer-end`
def one_liner(address: str, token: str, *, insecure: bool) -> str   # printf | curl ... | tail -c +6 | COLOSSEUM_HUB=... COLOSSEUM_TOKEN=... sh
def loader_main(argv: Sequence[str], *, environ=None, exec_fn=None) -> int   # stdlib + grpcio; Info -> bundle + uv env -> exec node agent
```

---

## Test support

| Module | What | Task |
|---|---|---|
| `tests/conftest.py` `_isolate_colosseum_dirs` (autouse) | `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `COLOSSEUM_SHM_DIR` → `tmp_path/{xdg-data,xdg-cache,shm}` | T1.3 |
| `tests/game_helpers.py` | `SP4_RUN`, `copy_sp4_run(tmp_path)`, `owner_sequence` (A4) | T0.1 |
| `tests/fixtures/sp4/` | SP4 run dir made by unchanged code + `make_sp4_run.py` + `manifest.json` (like `fixtures/sp3/`) | T0.1 |
| `tests/net_kit.py` | `hub_identity`, `node_identity`, `register`, `echo_handlers`, `loopback_server(credentials=None, *, interceptors=(), handlers=None)` (binds port 0), `channel(...)`, `call`, `call_stream` (A4) | T1.3 |
| `tests/hub_kit.py` | `FakeNode` (in-process client of the real hub server: register, session, send batches, read grants / acks), `MiniHubLoop` (a minimal loop over `hub.state` for T2.3), `make_result(...)` | T2.3 |
| `tests/node_kit.py` | `FakeHub` (in-process `Hub` servicer: scripted Welcome / grants / commands, records messages, serves blobs), `toy_node_options(tmp_path, ...)`, toy-game configs from `game_helpers` | T2.4 |
| `tests/cluster_kit.py` | in-process clusters (`start_hub`, `start_node`, `cluster`, …) and `hub_process(...)` over `cli_runner.start_hub`; `start_join` (T7.4) (A4) | T3.1 |
| `tests/fault_proxy.py` | `TcpFaultProxy(target=None)`: `set_target`, `drop()`, `delay(sec)`, `cut(sec)`, `stall()`, `restore()`; bound port 0 (A4) | T3.1 |
| `tests/cli_runner.py` | `hub_cmd`, `start_hub`, `wait_for_hub_json`, `hub_address` (T2.5); `tournament_hub_cmd` (T5.2); `node_env`, `join_cmd`, `start_join` (T6.3) (A4) | T2.5, T5.2, T6.3 |
| `tests/eval_kit.py`, `tests/join_kit.py`, `tests/ssh_kit.py` | as parts 05, 06, 08 define them (A4) | T5.1, T6.2, T8.1 |

Every module above goes into `known-first-party` in the task that creates it.

---

## Contract amendments

Binding (reconciliation of parts 01–08, 2026-10-18; controller decisions D1–D17 of `reconciliation-decisions.md`). They override the contract above and every part where they differ; a part's text that contradicts an amendment is wrong, and its brief says so. Vocabulary (D17): amendments **A1, A2, …** (here); controller rulings **R1–R18** and the accepted writers' rulings **W<part>-<n>** (numbered as in the part: part 01's "P1-1" → `W01-1`, part 02's "3." → `W02-3`, part 04's "P25" → `W04-25`) in the ledger; pre-flight rulings **P1, P2, …** in the SDD workspace. Every part's "Additions and deviations" not contradicted below are accepted; the ones other tasks rely on are restated here.

### Protocol

- **A1 (D1; T1.2 writes, every later task only uses).** `colosseum_v1.proto` is complete after T1.2: the outline above plus every row below (field numbers fixed here; the `Bootstrap` messages keep declaration-order numbers 1, 2, …; nothing is renumbered). No later task edits or regenerates the proto unless a need appears that no row covers (then it appends with the next free number and says so in its report); the "Modify: proto" lines of T2.3, T3.1, T3.2, T4.1, T5.1, T5.2, T7.1, T8.1, T8.2 are void (they extend T1.2's round-trip test only if they want). `PROTOCOL_VERSION` stays `(1, 0)`. T1.2's `test_sp5_messages.py` round-trips every row. T6.2 owns the test comparing the loader's own `Bootstrap` field numbers with the descriptors.

  | Message | Addition | First user |
  |---|---|---|
  | `Heartbeat` | `map<string, int64> counters = 6` (node-cumulative per session; names A2) | T2.4 |
  | `Heartbeat` | `repeated PathState paths = 7` | T3.2 |
  | `PathState` (new) | `int32 node = 1; string path = 2` (`direct` \| `relay`) | T3.2 |
  | `DataAddresses` (new) | `repeated string data_addresses = 1` | T3.1 |
  | `NodeMessage.m` | `DataAddresses addresses = 10` | T3.1 |
  | `Welcome` | `repeated string banned_node_ids = 12`; `bool no_workers = 13`; `int64 agents_version = 14` | T3.1 (filled T6.3); T5.2; T8.2 |
  | `Announce` | `repeated string banned_node_ids = 6`; `int64 agents_version = 7` (0 = unchanged); `string config_json = 8`; `repeated FixedBlob fixed = 9` | T3.1 / T6.3; T8.2 |
  | `Event` | `int64 generation = 9` (learner events; 0 otherwise) | T4.1 |
  | `Hello` | `repeated string leases = 16`; `string ssh_id = 17` | T5.1; T8.1 |
  | `BlockLease` | `int32 num_envs = 4` | T5.1 |
  | `BlockResult` | `string error_kind = 6` (`""` \| `abandoned` \| `crash` \| `missing_weights` \| `contract` \| `player` \| `config`) | T5.1 |
  | `BlockAck` (new) | `string lease_id = 1; repeated string block_ids = 2` | T5.1 |
  | `HubMessage.m` | `BlockAck block_ack = 7` | T5.1 |
  | `Command` | `repeated string lease_ids = 6` | T5.1 |
  | `LearnerAssignment` | `int64 start_env_steps = 8` (0 for agents of the first config) | T8.2 |

  Documented string values (comments in the `.proto`; `colosseum.net.messages.EVENT_KINDS` (T1.2) is the tuple of every `Event.kind`, the one validity list T4.4 checks): `Event.kind` = `proc_died`, `child_error`, `slot_disabled`, `learner_failed`, `warning`, `run_stopped` (T2.4), `node_stopped`, `node_leaving` (T4.4), `code_updated`, `code_update_failed` (T8.3). `Command.kind` = `start_learner`, `stop_learner`, `set_evaluators`, `release` (T6.3: stop as at the end of a run, exit 0), `stop_run`, `update_code`, `cancel_leases` (T5.1); `Command.reason` of `stop_learner` ∈ `superseded` (no final upload), `stop_run`, `freeze`, `remove` (final snapshot uploaded), `restart` (a recovery point instead, T8.3); of `stop_run` = `grace=<sec>`. `BlobRequest.kind` = `snapshot`, `fixed`, `teacher`, `init`, `latest`, `start` (T4.1), `eval` (T5.1), plus until T4.1 `recovery` / `trainer_state` (A28). `UploadHeader.kind` = `checkpoint`, `final`, `recovery`, `relay_latest` (T3.2), `agent_file` (T8.2, token-authenticated, A35). Refusals at registration are gRPC status, not messages (A33). Runtime floors `grpcio>=1.84`, `protobuf>=7.35.1`; `message Empty {}` serves `Bootstrap/Info`; `Bootstrap/Script` stays undeclared (R2).

### Counters, `hub_state.json`, test support

- **A2 (D2; T2.1 creates the book).** `colosseum.metrics.counters.CounterBook` (part 03 T3.1's code and `test_sp5_counter_book.py`, moved to **T2.1**) plus `by_source_prefix(prefix) -> dict[str, int]` (per name, the sum of the last cumulative values of every source whose key starts with `prefix`; per-node views use `f"{node_id}/"`). The hub owns `Hub._book` (T2.5; sources `"<node_id>/<session_id>"` from `Heartbeat.counters`, `"hub/<Hub._incarnation>"` from `RelayHub.stats()` on the metrics tick, and direct `add()` for hub events) and `MetricsHub.intake_book` (created by **T2.5** with `MetricsHub.on_intake_stats(agent_id, generation, depth, dropped)`, which `_on_metrics_batch` calls for every `IntakeStats`: depth → queue depths, `dropped` → `intake_book.report(f"learner/{agent}/{generation}", {f"dropped_chunks.{r}": n})`; T3.3 adds `stale` and `chunk_drops()`). Part 07's `counters["node_counters"]` / `["node_results"]` do not exist: per-node values are book sources and the per-node names below. The hub never derives a run total from events or a second dict. Canonical names (cumulative; dotted names nest one level in `grouped()`):

  | Name | Side (source) | Meaning | Task |
  |---|---|---|---|
  | `lineup_fetch_failed` | node | grants whose snapshot / blob / first-latest fetch failed | T2.4 (T3.1 across machines) |
  | `dropped_results_node_buffer` | node | results dropped by the full unacked buffer (= `Heartbeat.dropped_results`, kept; the hub reads only the counter). Part 07's `results_dropped_buffer` is this name | T2.4 |
  | `worker_restarts` | node | worker and evaluator restarts the node made (drained exits excluded, T8.2). The hub does **not** count `proc_died` events (T2.5's and T4.2's `worker_restarts += 1` are void) | T2.4 (policy T4.2) |
  | `results_sent` | node | results put into a `ResultBatch` the first time (resends not counted) | T7.1 |
  | `weight_fetches`, `weight_fetch_bytes`, `weight_fetch_failed`, `weight_fetch_not_modified`, `weight_fetch_redundant` | node | network fetches of latest weights (all agents) | T3.1 |
  | `weight_fetches/<agent>`, `weight_versions/<agent>` | node | the same per agent; distinct versions published from them (criterion 4) | T7.1 (counted at T3.1's fetch points) |
  | `data_port.latest_served`, `data_port.latest_not_modified`, `data_port.peer_refused` | node | `DataServer` | T3.1 |
  | `chunks_sent.direct`, `chunks_sent.relay`, `chunks_delivered`, `relay_frames_received` | node | router | T3.2 |
  | `dropped_chunks.{disconnect,no_learner,invalid,overflow}` | node (router) and hub (`hub/<inc>`: relay `overflow` / `disconnect`) | frames lost between machines | T3.2 |
  | `relay_frames`, `relay_bytes`, `relay_uploads`, `relay_uploads_refused` | hub (`hub/<inc>`) | relay | T3.2 |
  | `snapshot_holds`, `node_messages.invalid` | hub (`add`) | evicted-snapshot holds; malformed node messages (data addresses, …) | T3.1 |
  | `dropped_match_results_by_reason.{invalid,stale_generation,writer}` | hub (`add`) | intake drops, dead-writer results | T2.5 |
  | `results_journaled.<node number>` | hub (`add`) | results journaled (accounted) per node | T7.1 |
  | `learner_restarts` | hub (`add`, on every re-assignment after a died learner) | the node never restarts learners itself (W04-6) | T4.2 |
  | `lineup_redraw_failed` | hub | 16 rejected redraws in a row (A26) | T4.1 |
  | `dropped_uploads.{stale_generation,invalid,not_newer}`, `dropped_announcements.{stale_generation,not_newer,invalid}` | hub | | T4.1, T4.2, T4.4 |
  | `dropped_metrics.invalid`, `dropped_events.invalid` | hub | | T4.4 |
  | `dropped_chunks.{invalid,disconnect,stale,stale_generation}` | learner (`intake_book`, `learner/<agent>/<generation>`) | intake drops (`stale` T3.3; `stale_generation` real from T4.1) | T2.2 / T2.5, T3.3 |
  | `dropped_chunks_no_learner`, `dropped_chunks_disconnect`, `dropped_results`, `dropped_reward_episodes` | worker (`WorkerStats.counters` → `SystemStats` keyed (node, worker, incarnation); not a book) | | T2.2, SP4 |

  `system` mapping (T7.1): `dropped_chunks.<r>` = book + `intake_book` + workers' `dropped_chunks_<r>`, all six reasons present; `dropped_match_results_by_reason` = `{"node_buffer": book["dropped_results_node_buffer"], "invalid", "stale_generation", "writer"}`; `dropped_match_results` = workers' `dropped_results` + that dict's sum + the writer's `match_log.dropped` (part 07 note 7); `relay_bytes`, `lineup_fetch_failed`, `worker_restarts`, `learner_restarts` = book totals; `nodes_lost` = `HubStateData.nodes_lost` (a field, not a book name).

- **A3 (D2, D16; T2.1 / T2.5 write, T4.1 adds `learners`, T4.3 loads).** `hub_state.json`, the one schema: `{"version": 1, "run_id", "env_steps", "sessions": {"<node_id>/<session_id>": int}, "epoch_reserved_until": int, "versions": VersionRegistry.state() (parked infos never saved, A27), "counters": {"book": Hub._book.state(), "intake_book": MetricsHub.intake_book.state()}, "nodes_lost": int, "learners": LearnerTable.state(local_nodes=...)}` (`learners` from T4.1; absent before). `counters` is saved and loaded whole: unknown keys survive a resume (D16); T4.3 restores both books with `CounterBook.from_state`. Written atomically on every metrics tick and at the end (`fsync` only for epoch reservations, R7). Tests read totals through `cluster_kit.hub_counter(hub, name)` (book totals) or `read_hub_state(...)["counters"]["book"]["totals"]`; T2.5's `counters={"worker_restarts", …, "dropped_match_results": dict(self._drops)}` is replaced.

- **A4 (D3).** Test-support ownership (a later task that needs a knob a kit lacks adds it in its own task and reports it):

  | Module (owner) | Names |
  |---|---|
  | `tests/conftest.py` | `_isolate_colosseum_dirs` (T1.3); the `docker` skip (T7.3) |
  | `tests/game_helpers.py` | `SP4_FIXTURES`, `SP4_TTT_TINY`, `SP4_RUN`, `sp4_manifest`, `copy_sp4_run` and **`owner_sequence(workers, envs, agents, rounds)`** (T0.1; the baseline test and T2.1 / T2.5 import it from here, not from a test module); `queue_learner_io` (T2.2); `PulseFailBot` (T5.1) |
  | `tests/net_kit.py` (T1.3) | `hub_identity`, `node_identity`, `register(hub, node, ...)`, `echo_handlers`, `loopback_server(credentials=None, *, interceptors=(), handlers=None) -> (server, port)`, `channel(port, credentials=None, server_name=HUB_SERVER_NAME)`, `call`, `call_stream` (not `tls_channel`) |
  | `tests/hub_kit.py` (T2.3) | `FakeNode(address, *, token, ca_pem=None, insecure=False)`: `register`, `open(session_id, *, live_cells=(), last_acked=0)`, `send(**oneof)`, `cut`, `close`, `wait_for(kind, n=1, timeout=10)`, `download(kind, agent, ident)`, `upload(header, data)`; `MiniHubLoop`; `make_result`. Added by their first user (T4.1): `open(..., hello_fields=None)` (slots, hardware, learners, data addresses, name, tags), `pause_heartbeats()` / `resume_heartbeats()`, `wait_until(predicate, timeout)`; T6.3: `refuse_*` knobs |
  | `tests/node_kit.py` (T2.4) | `FakeHub(tmp_path, config, *, ttl_sec=1.0)` (`welcome`, `grant`, `hold_acks`, `serve_blob`, `not_found`, `send_command`, `announce`, `close_session`, `stop_server`, `restart_server`, `messages(kind)`, recorded uploads), `toy_node_options`; `run_node_main(options_json)` (T4.4); lease / ack scripting (T5.1). Part 03's `close_sessions()` and part 04's `drop_session()` mean `close_session()` |
  | `tests/cluster_kit.py` (T3.1) | `CLUSTER_TINY`, `wait_until`, `cluster_config`, `HubThread` (`hub`, `thread`, `root`, `address`, `token`, `insecure`, `exit_codes`, `stop()`), `start_hub(tmp_path, config, *, insecure=False, name="cluster")`, `NodeThread`, `start_node(...)`, `cluster(...)`, `read_hub_state(hub_or_root)`, `hub_counter(hub_or_root, name)`, `hub_version`, `journal_results_by_node`, and **`hub_process(config, tmp_path, name, *, overrides=None, args=())`** → `HubProc` (`address`, `token_file`, `root`, `env`, `wait(timeout)`, `interrupt()`, `tail()`; over `cli_runner.start_hub`; until T6.3 the token file is written from `HubIdentity.load_or_create(<the child's identity dir>).token`). `start_join(tmp_path, address, token_file, *, identity, args)` is added by T7.4 over the `cli_runner` join helpers. Part 04's `hub_in_thread` / `node_in_thread` / `.result()` mean `start_hub` / `start_node` / `stop()` + `exit_codes` |
  | `tests/fault_proxy.py` (**T3.1**) | `TcpFaultProxy(target=None, *, host="127.0.0.1")`: `set_target`, `delay`, `cut`, `drop`, `stall`, `restore`, `close`, `port`, `connections_seen`, `bytes_forwarded`; no `retarget` (T4.3 uses `set_target`) |
  | `tests/cli_runner.py` | T2.5: `hub_cmd(config, run_parent, name, overrides=None, *, port=0, insecure=True, listen="127.0.0.1", args=())` (a `None` / `False` keyword emits no flag), `start_hub(<same>, cwd=None, env=None) -> (Popen, run_root)` (own session), `wait_for_hub_json(run_root, timeout) -> HubInfo`, `hub_address(run_root, timeout=30.0) -> str`; T5.2: `tournament_hub_cmd`; T6.3: `node_env(tmp_path, name)`, `join_cmd(address, *, token_file, name, workers=1, evaluators=0, cpu_learners=1, once=True, extra=())`, `start_join(cmd, env, cwd=None)` |
  | `tests/eval_kit.py` (T5.1), `tests/join_kit.py` (T6.2), `tests/ssh_kit.py` (T8.1), `tests/report_kit.py` (`sample_report_data` with `nodes` and `training.lag`, T7.1) | as their parts define them |

### Hub process

- **A5 (D4; T2.5).** `hub.admin.call(target, command, args=None, *, timeout=30.0) -> {"code": int, "text": str, "data": Any}`; `hub.admin.stream(...)` yields the same dicts plus `"done": bool`; `resolve_hub(*, hub, run, token, token_file, insecure=False)`; `ADMIN_COMMANDS` gains `"hub.status"` (T2.5). The `hub` click group runs a hub only when invoked without a subcommand and requires `-c` only then (`hub token` needs no config). Admin data shapes: `nodes.list` / `nodes.remove` (T6.3, part 06), `nodes.add` and the `"ssh"` rows (T8.1), `tournament.*` (T5.2), `agents.*` (T8.2), `nodes.update_code` (T8.3) as their parts define.

- **A6 (D5, R5; T2.5 creates, later tasks extend).** Hub method names other parts rely on: `_on_message(ev)` dispatches each `NodeMessage` oneof to its own handler — `_on_heartbeat` (registry, budget, `self._book.report`), `_on_lineup_request`, `_on_result_batch`, `_on_metrics_batch`, `_on_version`, `_on_event`, `_on_data_addresses` (T3.1), the evaluation calls (T5.1; part 05's `_on_node_message` is `_on_message`); `_on_node_lost(entry, *, reason="lost", relocate_now=False)` (T2.5; T4.2 adds relocation; `node_leaving` (W04-24) and `nodes remove` (T6.3) call it with `relocate_now=True`); `_broadcast_announce()` (T2.5, the periodic `Announce`; part 03's "flush-style" sending) and `_announce_agents()` (T4.1, an immediate one); **`_fail_run(message: str, *, context: str = "") -> None`** (T2.5: ERROR, exit code 1, stop; replaces T2.5's `self._fatal = ...` assignments; used by reported child errors (W02-1), typed pulse block errors (T5.1), every-agent-failed (W04-27), unplaceable agents in `train` (W04-26)); **`_count_drop(node: int, group: str, reason: str, what: str, n: int = 1)`** (T2.5, replacing `_drop`: `self._book.add(f"{group}.{reason}", n)` + one WARNING per (node, group, reason); `_drop_writer` stays; T4.4 routes the remaining intake paths through it); `self._incarnation` (uuid4 hex per `launch`); `_node_rows()` (T6.3, A33), `_node_views()` / `_cluster_snapshot()` (T7.1), `_start_dashboard()` / `_stop_dashboard()` (T7.1), `_shutdown` / `_stop_nodes` / `_write_log_tail` (T4.4), `_resume_hub_state` / `_load_hub_state` / `_replay_journal` / `_resume_start_sources` (T4.3), `_on_tournament_blocks` / `_on_tournament_finished` (T5.2). `Hub.launch()` runs off the main thread (signal handlers installed only on the main thread), as `cluster_kit` needs.

- **A7 (D6; T1.2).** `VersionInfo` (frozen dataclass, contract fields) is defined in `colosseum.net.messages` with `to_dict()` / `from_dict()` (part 02 T2.1's code moves there); `colosseum.hub.state` imports and re-exports it and does not redefine it.

- **A8 (D7; T1.3).** T1.3 also creates `NodeBook.add_ephemeral(name, machine_id) -> (NodeRecord, secret)` (number 0, in memory, never in `nodes.json`; T2.5 no longer edits `security.py`), makes `HubIdentity.load_or_create` safe against two processes creating it at once (lock file or create-in-temp + atomic rename; tested), and puts `("grpc.so_reuseport", 0)` into `server_options`. Accepted part 01 notes 6–7: `NodeRecord`; `NodeBook.register(*, machine_id, name, tags, node_id, persistent=True, run_id="")` re-registers a known `node_id` with its number and name and a new secret, raises `SecurityError` for a banned node or machine; `NodeBook.{records, get, reload, set_cert_serial}`; `rotate()` keeps the same `NodeBook` object and moves `nodes.json` and `banned.json` away (so a rotation clears bans: W06-6 settled); `NodeIdentity.{key_pem, cert_pem, registered, hub_addresses, forget}`, `save_registration(reply, *, hub_addresses=())`, the node dir keeps `ca.crt`; `AuthInterceptor` refuses `NodeData/*` on the hub port; node data ports use `DataPortInterceptor(allowed: Callable[[str], bool])` (the contract docstring's "NodeData: …" clause is void); `grpc_util.{KEEPALIVE_MS, KEEPALIVE_TIMEOUT_MS, OPEN_METHODS, TOKEN_METHODS, token_metadata, node_metadata, metadata_value, node_id_of, peer_common_name}`.

- **A9 (part 02 note 11; T1.3, T2.5).** `train`'s local node is node number **0** (`add_ephemeral`), with `data_path="local"`, uids `<run_id>/train/n0/...` and cache / socket names `...-n0-...`; registered nodes are numbered from 1 and never reuse a number. `train --listen` keeps the local node at 0.

- **A10 (D10, part 02 notes 12–13, part 05 note 13; T2.5, T5.1).** `LocalNodeOptions.from_config(config, run_dir)` (T2.5): `workers = rollout.num_workers`, `evaluators` = 0 until T5.1 and **1 from T5.1 on** (one slot always; the hub runs it on demand: `Welcome.evaluators` / `set_evaluators` > 0 only while the pulse is on or a hub-run tournament is active, W05-2; R18's "1 iff `evaluation.enabled`" is revised), `log_dir = run_dir.logs`, `data_port = True` only with `train --listen` (→ `NodeOptions.data_port = -1`). `run_training(config_path, overrides=None, *, listen=None)` and `run_hub(config_path, overrides=None, *, listen=None, port=None, insecure=False)` in T2.5; `resume` (T4.3), `idle` (T5.2), `dashboard` (T7.1), `nodes_file` (T8.1) are added by their tasks, with `HubOptions.{resume, nodes_file, dashboard_port}` likewise. `HubOptions.bootstrap: bool = True` (T6.3): `False` builds no `BootstrapService` (the registration policy's `code_hash` is then `""`, so only source nodes are accepted); `cluster_kit.start_hub` passes `False` (T6.3 edits it) so in-process clusters never bundle the repository.

- **A11 (part 02 notes 14, 15, 17; part 03 assumption 3; T2.5, T3.1).** Until T4.1, `hub` mode places every learner on the first node that connects with learner capacity. Until T3.2 the hub strips `collect` from seats of agents whose learner is not on the requesting node (it grants such a node non-collecting lineups; part 02 note 14's "no lineups" is replaced by this, which part 03's T3.1 tests need). `hub.registry`: `HubInfo.loopback_tls`, `to_json`, `from_dict`, `this_process_info`, `alive`, `RUNNING_DIR`. `MatchLogWriter.dead`; a tokened `submit` refused by a full writer is never counted in `WriterStats.dropped`. `Coordinator.lineup_for(owner)`, `save_league_state(path, *, env_steps, journal_row=None)` are T2.5's.

- **A12 (W02-1, W04-25, part 02 note 18, part 04 P6; T2.4, T2.5, T4.2).** A worker or evaluator that reports a `ChildError` stops the run (exit 1) through `_fail_run`; only deaths without a report (signal, OOM) restart, by the node, within `WORKER_RESTARTS`, counted in the node's `worker_restarts` (A2). A learner death stops the run until T4.2; from T4.2 the node reports `proc_died` / `learner_failed` with `Event.generation` and the hub re-assigns (new generation, start point A28), counting `learner_restarts`; no learner restart while the hub is away. Ported SP2 tests follow W04-25.

- **A13 (D11; T1.1, T7.4).** Criterion 5 passes by W01-1 (bytes → checked numpy payload, ratio of medians ≥ 10 against today's lz4 wire); `scripts/measure_frame_decode.py` also prints the full path to a built `TrajectoryChunk` for both sides (old: unpack + `validate_chunk_payload` + `from_payload`; new: frame decode + checks + `from_payload`); T7.4's rerun and the acceptance report show both numbers.

- **A14 (D12; T2.5, T7.6).** No `test_sp5_two_trains_slow.py`. T2.5 adds the fast contract test `tests/integration/test_sp5_two_hubs.py::test_two_train_hubs_on_one_machine_do_not_collide` (two `Hub` objects in `train` mode in one test, with idle in-process nodes or none, no loaded workers or learners): distinct ephemeral loopback ports, distinct `hub.json` and running-registry entries, non-colliding run-scoped shm and socket prefixes, both serving sessions at once. T7.6's runbook for criterion 6 adds a step: two real `colosseum train` at once on the quiet machine, outside the suite.

- **A15 (D13).** Behaviour changes of SP1–SP4 the acceptance report lists for the owner: W02-1 (reported Python exceptions of workers stop the run), W04-25 (restarts replace "a dead child fails the run"), W04-15 (only the newest final of the current run id is protected), W05-1 (the pulse backlog drops only snapshots not started), W08-4 (`update-code` keeps the node agent process and spawns children on the new bundle), part 08 note 10 (`Hub/Upload` accepts the token for `agent_file` only), W06-5 (the token on stdout, redacted in `main.log`), W07-1 (dashboard as a hub child process), W01-2 (no mask-legality check of recorded actions).

### Frames, chunks, caches

- **A16 (D14, part 01 note 1; T1.1).** Checkpoint frames carry exactly the meta keys `agent_id`, `policy_version`, `final`, `kind` (`periodic` \| `final` \| `recovery`), `generation`; `decode_checkpoint` refuses any other key set; readers (T2.5 `_on_upload`, T4.2 `RecoveryStore`, T4.1 start blobs) use these names. Constructors: `FrameError(reason, message="")`, `ChunkError(reason, detail="")` (part 04's one-argument calls are valid). Frame additions: `decode_frame(buf, *, cap, copy=False, kind=None)` (`copy=True` = one private copy of the data region with writable views), `frame_sizes(buf) -> (raw, stored)` (for `Lz4Probe.observe`), `lookup_class`, `Lz4Probe.enabled`, `MAX_NDIM`, `FLAG_LZ4`, `FRAME_KINDS`, `DTYPES`; NaN / Infinity refused in headers and scalars.

- **A17 (part 01 notes 2–3, W01-2; T1.1, T2.2, T3.3).** `ChunkValidator(role_spec, *, agent_id, teacher, chunk_length=None, state_template=None)` with `check(payload, agent_id, policy_version=None)` (numpy payload) and `validate(chunk)` (built chunks; queue path, tests); `decode_chunk(buf, *, cap, validator=None)` runs frame → checks → chunk and raises `ChunkError("invalid", "frame:<reason>: …")` for frame errors too; `chunk_meta`, `check_payload_fields`, `WEIGHTS_KINDS`, `CHECKPOINT_KINDS`; `core.types.check_slot_arrays(...)` (T1.1 edits `core/types.py` for it, A36). **`ConnectionIntake._accept` (T2.2) is: `peek_meta` → misroute and generation rule on the header (T3.3 adds the lag rule there) → `decode_chunk(frame, cap=…, validator=self._validator)`**; it never calls `validator.validate` on a built chunk. `node.learner_main` builds the validator with `chunk_length=config.rollout.chunk_length` and `state_template=state_to_numpy(model.initial_state(1))`.

- **A18 (part 01 note 9, part 02 note 7, part 05 note 7, part 03 note 11; T1.5).** Cache keys of hub blobs: `colosseum.node.weights.blob_key(kind, agent_id, blob_id) -> CacheKey("blob", agent_id, f"{kind}-{blob_id}")` for `kind` ∈ `fixed`, `teacher`, `init`, `eval`, and `split_blob_key(key) -> (kind, blob_id)` (the `BlobRequest` of a key); T1.5's blob ident regex becomes `^(fixed|teacher|init|eval)-[A-Za-z0-9_][A-Za-z0-9_.-]{0,159}$` and the file `{P}-blob-{ident}`. T2.4's `f"{kind}-{blob_id}"` is this rule; T5.1's eval blobs use `blob_key("eval", agent_id, blob_id)` (`player_ref_to_proto` sends the raw `blob_id`, `player_ref_from_proto` rebuilds the key). Also T1.5: `CacheLockedError`, `release_node_lock`, `segment_name`, `pointer_name`, `frame_digest`, `KEEP_LATEST`, `READ_RETRIES`, `SPACE_MARGIN_BYTES`; T3.1: `WeightCacheReader.open_latest(agent_id) -> (BinaryIO, pointer dict) | None`.

- **A19 (part 01 note 8, part 03 notes 13, 19; T1.4, T3.3).** `translate_transport(data)` with one WARNING per load; `TRANSPORT_IGNORED_KEYS`, `TRANSPORT_RENAMED_KEYS`; `DistributedConfig` checks `node_lost_after_sec > heartbeat_sec` and `learner_relocate_after_sec >= node_lost_after_sec`; `learner.device` ∈ `auto|cpu|cuda|cuda:N|mps`. T3.3 adds the `LearnerConfig` check `max_policy_lag < weight_push_interval − 1` → `ConfigError` (and edits `core/config.py` for it).

### Hub state, local connections, server, node

- **A20 (part 02 note 1, part 03 note 6, part 04 notes 3, 12, part 08 note 8; T2.1 and the named tasks).** `hub.state` names others use: `NodeHello` (T2.5's `hub.server.hello_from_proto` converts), `NodeRegistry.connect(..., *, data_path="relay")`, `disconnect`, `remove`, `by_id`, `all`; `EpochAllocator(reserved_until, on_reserve, *, block=EPOCH_BLOCK)`, `reserved_until`; `CellTable(epochs)`, `grant`, `owners(exclude=)`, `drop_cells`, `get`, `drop_worker` (T4.2); `restrict_collecting`; `BudgetCounter.add`; `ResultIntake.check(result, *, spec, agents, versions, node=None, uid_prefix=None)`, `accounts_at_intake`, `count_drop`, `dropped`, `open(node_id, session_id, last_acked)`, `is_acked`, `forget_node`; `DROP_REASONS`; `version_source`, `SnapshotHolds`, `snapshot_refs` (T3.1); `collect_allowed`, `strip_collect` (T3.2); `VersionRegistry.park` / `unpark` / `parked` (T4.1); `OwnerRotation.set_agents` (T8.2). `OwnerRotation` reproduces SP4's sequence for simultaneous expiry only (W02-2).

- **A21 (part 02 notes 3–5, part 03 note 14, part 05 note 8, part 08 note 11; T2.2 and the named tasks).** `net.local` additions: `ControlChannel` (`ConnectionIntake` and `NodeOutputs` take it instead of a raw `Connection`), `FrameSender`, `listener_socket`, `RUNTIME_DIR_ENV`, `MAX_SOCKET_PATH`, `SEND_SLICE_SEC`, `PROGRESS_INTERVAL_SEC`, `INTAKE_REPORT_SEC`; `WorkerIO` gains `pump`, `take_lineups`, `idle`, `counters`, `send_progress`, `send_error`, `close`, `stats_fn`, `dropped` and keywords `results_batch`, `results_flush_sec`, `clock`; `LearnerOutputs.checkpoints`, `finish(stop_event)`; `ConnectionIntake.value`, `close`, `stop_reason` (T4.1), `max_policy_lag=None` (T3.3); `ChunkIntake.note_policy_version` (T3.3); `EvalTask`, `EvalCancel`, `EvalDone`, `EvalFailed` (T5.1); `Drain(reason, timeout_sec)` (T8.2). `RolloutLoop(..., snapshot_fallback=True)` (node workers pass False); `learner.learner.set_policy_version`, `_publish_weights` (replaces `_push_weights`), `lag_metrics` (T3.3). `learner_process` keywords: `version_floor`, `recovery_interval_sec` (T2.2), `send_final: Callable[[], bool] | None = None` (T4.1), `clock=time.monotonic` (T4.2); `utils.seeding.learner_seed(seed, agent_index, generation=1)` (T4.1).

- **A22 (part 02 notes 6–7, part 03 notes 5, 8, 13, 16, part 04 notes 3–4, part 05 note 7, part 06 notes 9, 11, part 08 note 10).** `hub.server`: `build_server(*, inbox, identity: HubIdentity, insecure, listen, loopback_plain, blobs, max_message_mb, bootstrap=None, relay=None, latest=None, policy=None)` (identity required; `relay` / `latest` T3.2, `policy` T6.3; T6.2 adds `BootstrapServicer` / `add_bootstrap` / `BOOTSTRAP_SERVICE` and one registration line); `LOOPBACK_KEY`, `SERVER_THREADS`, `OUTBOUND_QUEUE_SIZE`, `UPLOAD_REPLY_TIMEOUT_SEC`, `REGISTER_REPLY_TIMEOUT_SEC`, `StopRequested`, `NodeMessageIn.results` / `error`, `SessionOpened.warnings` (T6.3); `RelayEvent` is not used (relay counters come from `RelayHub.stats()`, A2). `BlobStore` (one class, methods by task): `frame_path(kind, agent_id, ident)` (T2.3; snapshots fall back to `checkpoints/.evicted/`, T3.1), `register_fixed(blob_id, frame, *, kind, agent_id)` and `blob_ident(kind, agent_id, data)` (T2.3), `forget(kind, agent_id, ident)` (T3.1), `parked_path(agent)` = `blobs/latest-<agent>-parked.frame` and `start_path(agent, ident)` = `blobs/start-<agent>-<ident>.frame` (T4.1), `register_weights(blob_id, weights_file, *, agent_id, networks, roles)` / `unregister(blob_id)` (T5.1); files `runs/<name>/blobs/<kind>-<agent>-<ident>.frame` (`trainer_state`: `.bin`). `Download` dispatch: `snapshot`, `fixed`, `teacher`, `init` (T2.3), `recovery` / `trainer_state` (T2.5, until T4.1), `latest` (parked first, T4.1; else relayed, T3.2), `start` (T4.1), `eval` (T5.1). `Upload`: `checkpoint` / `final` (T2.5) and `recovery` (T4.2) post `UploadIn`; `relay_latest` is handled in the servicer by `LatestRelay` (T3.2); `agent_file` (T8.2). `grpc_util`: `send_parts(data: bytes | memoryview | Path, *, meta_json="", digest=None)`, `receive_parts` returns `(None, meta)` for `not_modified`, `BlobError`, `blob_digest`, `not_modified_part` (T2.3).

- **A23 (part 02 notes 9–10, part 03 notes 9–10, part 04 notes 11, 18, part 06 note 12, part 08 note 2).** `NodeOptions` fields (frozen dataclass; contract fields plus): `credentials: tuple[str, int, str] | None = None`, `shutdown_grace_sec: float | None = None` (T2.4); `data_addresses: tuple[str, ...] | None = None` and `data_port` meaning `None` none / `0` ephemeral / `-1` the hub config's `distributed.data_port` (`join`'s and `train --listen`'s default) (T3.1); `code_hash: str = ""`, `bundle_dir: Path | None = None` (T6.3). `NodeAgent(options, *, exec_fn=os.execve)` (T6.3); accessors `counters()`, `fetch_history()` (T3.1), `paths()`, `data_port()` (T3.2); `ResultBuffer`, `CellTracker`, `LINEUP_RETRY_SEC`, `METRICS_FLUSH_SEC`, `LOCAL_CHILD_GRACE_SEC` (T2.4), `PENDING_EVENTS_MAX = 100` (T4.2), `join_slots`, `build_agent_parser`, `options_from_args`, `join_argv`, `_reexec_loader`, `_switch_run` (T6.3); env vars `COLOSSEUM_SSH_ID` / `COLOSSEUM_SSH_SESSION` (T8.1). `node.session`: `SessionUp`, `SessionLost`, `backoff_delays`, `hello` is a callable (T2.3); `HubSession.away_for()`, `slow_retry(on)` (T4.4); `SessionRefused`, `SessionUnauthenticated`, `SessionBanned`, `SessionKeyMismatch` (T6.3). `learner.factory.teacher_from_frozen`, `init_from_blob` (T2.4).

- **A24 (D8, part 02 note 10, part 04 note 17, part 08 notes 11, 15).** `node.hardware`: probes `affinity_cores`, `cgroup_quota_cores`, `physical_cores_raw`, `nvidia_gpus`, `torch_versions`; `Hardware.torch_version` / `torch_build` come from `colosseum.bootstrap.torch_select.torch_identity()` (T6.1), so hub and node compare the same strings. `node.procs.ChildSupervisor`: `start(spec) -> int` (T8.3 adds `code: ChildCode | None = None`), `exited`, `may_restart`, `stop_all`, `get`, `incarnation`, `names`, `sentinels`, `restart`, `forget` (T2.4; the target gets `incarnation`); a drained exit (code 0 while draining) is neither a failure nor a restart (T8.2); `ChildCode`, `child_code`, `DRAIN_TIMEOUT_SEC` (T8.2 / T8.3); `watch_parent` re-exported from `utils.process` (T4.4: `PARENT_WATCH_SEC`, `start_parent_watchdog`, `watch_parent`); `core.run_state.process_start_ticks` portable (T4.4).

- **A25 (D15, part 03 notes 2–5, 7, 21; T3.1, T3.2).** New modules: `colosseum.node.fetch` (`FetchPlanner`, `FetchJob`, `FetchStats`) and `colosseum.node.dataport` (`DataServer` with `start(port, credentials=, host=) -> int`, `stop`, `set_node`, `counters`; `probe_direct`, `fetch_latest`, `fetch_latest_relay` (T3.2), `read_latest_reply`, `check_fetched`, `FetchError`, `FetchedLatest`, `data_addresses`, `frame_digest`, `DATA_PORT_TRIES`, `DATA_SERVER_THREADS`) in **T3.1**; `DataServer.Chunks` (T3.2). `colosseum.node.router` (T3.2) holds only the outbound path (`ChunkRouter`, `RouteTable`, `RemoteRoute`, `TargetSender`, `RouteReader`, `LocalDelivery`, `RelayReceiver`, `Counters`, constants); the contract's `node.router.DataServer` is void. `colosseum.hub.relay` (T3.2: `RelayHub`, `InboxHandle`, `LatestRelay`, `RelayedVersion`, constants). `CheckpointManager.{set_eviction_hold, held_dir, release_held, clear_held}`, `EVICTED_DIR` (T3.1); `.evicted/` is cleared at every hub start.

### Learners, failures, resume

- **A26 (part 04 notes 1–2, 6, 8; T4.1).** `hub.placement`: `NodeCapacity.mps: bool = False`; `PlacementPlan.new(current)`; `parse_device`, `requested_slot`, `capacity_from_hello(node, name, tags, slots, gpu_kinds, *, unlimited)`; `LearnerTable` / `LearnerEntry` / `Reconciled`, `LEARNER_STATUSES` (`pending`, `starting`, `running`, `relocating`, `failed`), `TABLE_STATE_VERSION = 1`, `CPU` / `CUDA` / `MPS`; `place(...)` keeps the contract signature; `describe_plan` / `describe_distributed` are T7.2's. `LearnerTable.can_collect` is true for `starting` and `running` (W04-5). `colosseum.hub.hub.LINEUP_REDRAWS = 16` (W04-4). `train` records gain `learner_node`, `generation`; `system` gains `pending_agents` (T4.1) and `failed_agents` (T4.2).

- **A27 (part 04 note 3; resolves a conflict with A9).** A parked `VersionInfo` (an agent without a learner, played with hub-held weights) has **`node = -1`** (not 0: node 0 is `train`'s local node, whose `FetchPlanner` skips versions of its own node number), `generation = 0`, `relay_only = True`, no addresses; `is_stale(agent, 0 | None, v)` is False; it is never saved in `hub_state.json`; `Download(kind="latest")` serves `BlobStore.parked_path(agent)` (with `since_version` → `not_modified`).

- **A28 (part 04 note 4; resolves T2.5 vs T4.1).** Learner start blobs. Until T4.1, T2.5 names a `training.resume_from` state with `LearnerAssignment.weights_blob` = a blob id of kind `recovery` (the model frame) and `trainer_state_blob` = a blob id of kind `trainer_state` (`.bin`), and an `init` state through `Welcome.fixed` (`kind="init"`). From T4.1 on, `weights_blob` ∈ {`""`, `init`, `start:resume`, `start:recovery`, `start:ckpt_v<N>`}, every `start:*` blob is one checkpoint frame (`encode_checkpoint(..., kind="recovery", generation=0)` for synthesized starts, or the stored recovery point) fetched with `BlobRequest(kind="start", ident=<resume|recovery|ckpt_v<N>>)`, `trainer_state_blob` is always `""`, and T4.1 removes the `recovery` / `trainer_state` download kinds and T2.5's resume-state blobs. The version floor is max(registry floor, newest stored snapshot + 1, recovery point + 1) on the hub and the node's cached latest + 1 (W04-7).

- **A29 (part 04 notes 10–11, 13–16, 20, part 01 assumption T4.3, part 08 assumption 5; T4.2, T4.3).** New modules `colosseum.hub.recovery` (`RecoveryStore(run_root)`, `RecoveryPoint`, `read_frame_meta`, `RECOVERY_DIR` / `RECOVERY_FRAME` / `RECOVERY_META`; `RunDir.recovery_dir`) and `colosseum.node.uploads` (`UploadQueue`, `Upload`, `UPLOAD_CONCURRENCY = 2`: every checkpoint frame of a learner goes through it, finals never dropped) (T4.2); `colosseum.hub.resume` (`ResumePlan`, `prepare_resume`, `commit_resume`, `check_not_running`, `check_resumed_agents`, `config_diff`, `flatten`, `next_backup_path`, `start_env_steps`, `resume_port`, `AGENTS_STATE_FILE`, `DIFF_LINES_MAX`, and the hook `_apply_agents_state` that T8.2 fills) (T4.3). Journal: `READABLE_SCHEMA_VERSIONS = (1, 2)`, `meta.next_match_id` (row ids never reused), `MatchRow.node`, `MatchLogReader.results_after`, `folded_after`, `last_row`, `max_env_steps` (reader and writer). Coordinator: `LeagueStateInfo`, `league_replay_window`, `loaded_league_state`, `replay_results(results) -> (replayed, skipped)` next to `league_state_journal_row()`. Retention protects only the newest final of the current run id (W04-15). `hub --resume` without `--port` listens on the previous `hub.json` port. An in-place resume drops the saved config's `transport` key before applying `--set`.

- **A30 (part 07 note 5; T4.4, T7.1).** The node's stop order is: stop children → flush results → wait for acks within the grace → **a final `Heartbeat`** (T7.1 adds it to T4.4's stop path) → `Event(kind="run_stopped")` → close.

### Evaluation, joining, metrics, part 08

- **A31 (part 05 notes 1, 3–6, 8–12, 14; T5.1, T5.2).** `PlayerRef` and `BlockTask` live in `colosseum.net.messages` (re-exported by `hub.evaluation`); `BlockTask.lineups` / `players` are tuples, plus `num_envs: int = 8`; `PULSE_PRIORITY = 1`, `TOURNAMENT_PRIORITY = 0` and the `player_ref_*` / `block_task_*` / `lease_to_proto` converters. Lease protocol: `BlockLeaseRequest.free_slots` is an increment; one `BlockLease` per grant, round robin; every `BlockResult` is acked (`BlockAck`) and resent until acked; `Hello.leases` names held leases. `BlockQueue.complete(...)` returns `"accepted" | "late" | "cancelled" | "unknown"`, `cancel(owner) -> (block ids, {node: [lease ids]})`, plus `want`, `unwant`, `waiting`, `dispatch`, `release`, `keep_only`, `forget`, `take_failed`, `task`, `leases_of`, `queued`, `live`, `stats`; `MAX_BLOCK_ATTEMPTS = 3`. `HubEvaluation`, `HubPulse` (replaces `PulseFeeder.pump`; `PulseRequest.weights: str`; `pulse_target`, `pulse_main`, `PulseResults` deleted), `colosseum.node.leases`, `evaluator_main.{evaluator_loop, CachePlayerPool, MissingWeights, play_task, CONNECT_TIMEOUT_SEC, POLL_SEC}`; engine renames `MatchPlacement`, `error_kind`. Run-dir `eval/pulse/<agent>/<ckpt>/`, `hub_tournaments.json`, `tournaments/<name>/submitted-config.yaml`; `evaluator-<i>.log` replaces `pulse.log`. CLI `tournament --hub ADDR|RUN_DIR [--token T] [--token-file F] [--cancel NAME]`. A typed evaluator error is never a `ChildError`; a typed error in a pulse block calls `_fail_run` (A6).

- **A32 (D9, part 06 notes 1–8; T6.1, T6.2, T8.1).** `torch_spec(hub_torch, hub_build, *, platform: NodePlatform, gpus: Sequence[NvidiaGpu]) -> TorchChoice(requirement, backend, cuda_driver, warning)`; `installer_script(*, insecure, python, grpcio, uv_version=UV_VERSION) -> bytes` (address from `COLOSSEUM_HUB`, token from `COLOSSEUM_TOKEN`, extra arguments to `join`); `one_liner(address, token, *, insecure) -> str` (`... | COLOSSEUM_HUB='<addr>' COLOSSEUM_TOKEN='<token>' sh`); `join_command`; `loader_main(argv, *, environ=None, exec_fn=None)` (self-contained: stdlib + grpcio, own `Bootstrap` codec; `COLOSSEUM_NODE_PYTHON`, `COLOSSEUM_UV`). `Bundle(path, code_hash, files, size, python_path=(), skipped=())`, `build_bundle(..., roots=None)`, layout `bundle.json` + `project/` + `pkg/<i>/`; `requirements.EXCLUDED` adds `cuda-*`, `pytorch-triton*` (and local-path distributions); environments `envs/<key>/`, `key = sha256(requirements_hash|python|torch_version|backend)[:24]`; `BootstrapService(...)` + `make_bootstrap_service(...)`. T8.1 runs the installer as `ssh ... 'sh -s -- <join flags>'` with `COLOSSEUM_HUB=127.0.0.1:<remote port>` and `COLOSSEUM_TOKEN` exported by its stdin preamble (part 08 assumption 1 adapts).

- **A33 (part 06 notes 10–13, part 08 notes 3, 16; T6.3, T8.1, T8.3).** New modules `colosseum.net.registration` (`RegistrationPolicy`, `Refusal`, `parse_refusal`, `check_hello`, `need_bytes`, `is_loopback_peer`, `REFUSAL_REASONS`, `CACHE_HEADROOM_BYTES`) and `colosseum.hub.joining`. Refusals are gRPC `FAILED_PRECONDITION` with JSON details `{"refused", "message", "code_hash"}` before `Welcome`; an empty `Hello.code_hash` is accepted only from a loopback peer (W06-2). `Hub._node_rows()` (T6.3) is the one base row of a node — `{"number", "name", "node_id", "state": online|lost|removed|offline, "hardware", "roles", "data_path", "last_heartbeat_age_sec", "tags", "code_hash"}` — used by `nodes list` and, extended (A34), by `nodes.json`; T8.1 adds `"ssh"` rows to `nodes.list` data. `_reexec_loader()` is used only for a code-hash refusal at (re)connect; `Command(update_code)` does not re-exec the agent (W08-4; part 06 assumption 10 is void).

- **A34 (part 07 notes 1–3, 6–9, 11–12, 15–18, part 03 note 18, part 05 note 11; T7.1, T7.2, T7.3, T7.4).** `colosseum.metrics.nodes` (T7.1) builds `nodes.json` rows from `Hub._node_rows()` plus `env_steps_per_sec`, `results_per_sec`, `load`, `env_steps`, `results_journaled` (book `results_journaled.<n>`), `results_sent`, `results_dropped` (`dropped_results_node_buffer`), `weight_fetches` (`{agent: {"fetches", "versions"}}` from `weight_fetches/<agent>`, `weight_versions/<agent>`); `by_node` cells gain `state`. `REQUIRED_KEYS["system"]` is extended **by T7.1** with `nodes`, `nodes_lost`, `by_node`, `relay_bytes`, `dropped_chunks`, `dropped_match_results_by_reason`, `lineup_fetch_failed`, `worker_restarts`, `learner_restarts`, `pending_agents`, `failed_agents`, `live_ratings`, `evaluation` (`HubEvaluation.stats()`; its cumulative fields count since the hub process started — leases are not persisted, W05-5); `system` also carries, not required, `dropped_uploads`, `dropped_announcements`, `dropped_metrics`, `dropped_events` (book groups) and `lineup_redraw_failed`. `REQUIRED_KEYS["train"]` is unchanged (`policy_lag_p50` / `_p95` / `_max`, `dropped_stale_chunks`, `learner_node`, `generation` are always written but not required). `SystemStats.snapshot(..., cluster=None)`, `MetricsHub.maybe_tick(..., cluster=None)` / `close(..., cluster=None)`, `ConsoleReporter.nodes_line`; report `SECTIONS` gains `"nodes"` after `"system"`, training `"lag"`; `dashboard_child(run_dir, port, conn)`; `hub --dashboard [PORT]` only on `colosseum hub`. T7.2: `utils.logging.setup_process_logging(..., keep_until_file=False)`, `PendingRecords`, `flush_pending`, `discard_pending`, `PENDING_LIMIT`, the node's pre-`Welcome` log `<cache_home>/colosseum/logs/<ca8>/join.log`, `cli.REMOVED_COMMANDS` (hidden commands, exit 2), `placement.describe_distributed`. T7.3: `deployment/node-entrypoint.sh`, targets `node` / `hub`, the `docker` marker implies `slow`. T7.4: `scripts/measure_common.py`; `node.learner_main.algorithm_factory` / `node.worker_main.collect_chunks` only if missing. The slow-suite count is what the suite has after T7.4 (T7.5 writes the real number; part 07 note 18's "16" is void).

- **A35 (part 08 notes 1–15, 17).** New modules `colosseum.hub.ssh` (T8.1), `colosseum.hub.agents`, `colosseum.hub.jobs` (T8.2), `colosseum.hub.update`, `colosseum.bootstrap.codecheck` (T8.3), `tests/ssh_kit.py`. `Membership` (in `league.base`, exported by `colosseum.league`) and the `membership=` keyword of `MatchmakerContext.from_config`, `validate_matchmaking`, `describe_mix`, `validate_config` (train scope), `Coordinator(...)`; `MatchmakerContext.start_env_steps`, `owner_env_steps(owner)`; `BaseMatchmaker.on_agents_changed(context) -> BaseMatchmaker` (returns the matchmaker to use); `Coordinator.on_agents_changed(config, player_roles, membership)`, `Coordinator.membership`; `PfspStats.owners` / `add_owner` / `remove_owner`, `RatingBook.agent_ids` / `add_agents`, `LiveRatings.set_agents`, `MetricsHub.agent_ids` / `set_agents`; `RolloutLoop.episode_counts()`; `AGENT_VALIDATION`, `AGENT_VALIDATION_TIMEOUT_SEC`; `agents_state.json` as T8.2 defines; `AuthInterceptor` (T8.2 edits `grpc_util`) lets `Hub/Upload` authenticate with the token secret, and the servicer then accepts only `UploadHeader.kind == "agent_file"` (part 08 note 10); `bootstrap.bundle.member_for`; `SshLink.finish()`; the K8s `Namespace` lives in `hub.yaml`.

### Corrections to the file structure and the task index

- **A36 (index; last).** The «File structure» and «Task index» above change as follows (main files; parts list the rest):

  | Task | Add / change |
  |---|---|
  | T0.1 | M `.gitignore`; `owner_sequence` in `tests/game_helpers.py` (A4) |
  | T1.1 | M `core/types.py` (`check_slot_arrays` only; A17); the criterion-5 script prints both rules (A13) |
  | T1.2 | the complete proto (A1); `EVENT_KINDS`, `VersionInfo` with `to_dict` / `from_dict` (A7) |
  | T1.3 | `NodeBook.add_ephemeral`, race-safe `HubIdentity.load_or_create`, `so_reuseport` (A8) |
  | T1.4 | M `tests/unit/test_bench_throughput.py` |
  | T1.5 | integrated after T1.1 (imports `net.chunks`); `blob_key`, `split_blob_key` (A18) |
  | T2.1 | C `src/colosseum/metrics/counters.py` (`CounterBook` + `by_source_prefix`), `tests/unit/test_sp5_counter_book.py` (moved from T3.1; A2) |
  | T2.2 | M `tests/game_helpers.py` (`queue_learner_io`) and the four ported learner test files (part 02); intake order (A17) |
  | T2.3 | no proto change (A1) |
  | T2.4 | M `learner/factory.py`; uses `bootstrap.torch_select.torch_identity` (A24) |
  | T2.5 | M `metrics/hub.py` (`on_intake_stats`, `intake_book`), `tests/cli_runner.py` (A4 names); no `net/security.py` edit (A8); test `tests/integration/test_sp5_two_hubs.py` instead of `test_sp5_two_trains_slow.py` (A14) |
  | T3.1 | C `node/fetch.py`, `node/dataport.py`, `tests/fault_proxy.py`; no `metrics/counters.py`, no proto; M `node/weights.py`, `hub/state.py`, `coordinator/checkpoint_manager.py`, `pyproject.toml` |
  | T3.2 | C `hub/relay.py`; M `node/dataport.py` (`Chunks`), `hub/state.py`; no `tests/fault_proxy.py`, no proto |
  | T3.3 | M `node/learner_main.py` (one keyword), `core/config.py` (one validator) |
  | T4.1 | M `hub/state.py`, `hub/server.py`, `learner/learner.py`, `utils/seeding.py`; no proto |
  | T4.2 | C `hub/recovery.py`, `node/uploads.py`; M `node/learner_main.py`, `hub/server.py`, `hub/state.py`, `core/run_dir.py`, `tests/integration/test_sp2_lifecycle.py` (`node/procs.py` only if the restart policy needs it) |
  | T4.3 | C `hub/resume.py`; M `arena/retention.py` |
  | T4.4 | M `node/logs.py`, `tests/node_kit.py` |
  | T5.1 | C `node/leases.py`, `tests/eval_kit.py`; M `net/messages.py`, `net/local.py`, `arena/tournament/engine.py`, `core/config.py`, `hub/server.py`, `tests/game_helpers.py`, `pyproject.toml`; ported `test_sp4_pulse.py`, `test_sp4_pulse_runs.py`; no proto |
  | T5.2 | M `node/agent.py` (`no_workers`), `tests/cli_runner.py` (`tournament_hub_cmd`); no proto |
  | T6.2 | C `tests/join_kit.py`; M `pyproject.toml` |
  | T6.3 | C `net/registration.py`, `hub/joining.py`; M `hub/server.py`, `node/session.py`, `tests/hub_kit.py` / `tests/node_kit.py` knobs, `tests/cluster_kit.py` (`bootstrap=False`, A10) |
  | T7.1 | C `metrics/nodes.py`; M `node/agent.py`, `tests/report_kit.py`; no proto |
  | T7.2 | M `utils/logging.py`, `hub/placement.py`, `hub/hub.py` (start-up only), `node/agent.py` (start-up logging) |
  | T7.3 | C `deployment/node-entrypoint.sh`; M `tests/conftest.py`, `tests/unit/test_example_configs.py` |
  | T7.4 | C `scripts/measure_common.py`, `docs/benchmarks/sp5-*`; M `tests/cluster_kit.py` (`start_join`); maybe `node/learner_main.py`, `node/worker_main.py` |
  | T7.6 | the two-`train` runbook step (A14) |
  | T8.1 | C `hub/ssh.py`, `tests/ssh_kit.py`; M `node/agent.py`, `pyproject.toml`; no proto |
  | T8.2 | C `hub/agents.py`, `hub/jobs.py`; the part 08 file list; no proto |
  | T8.3 | C `bootstrap/codecheck.py`, `hub/update.py`; M `hub/admin.py`, `node/procs.py`, `node/evaluator_main.py`, `node/learner_main.py` |

  Parallel splits after these changes: **group A** — `core/types.py`: T1.2 the three dataclass fields, T1.1 only `validate_slot_structure` / `check_slot_arrays` (disjoint hunks); `pyproject.toml` `[project].dependencies` is touched by T1.2 (grpcio, protobuf) and T1.3 (cryptography): the controller merges the list at cherry-pick; T1.5 is developed on and integrated after T1.1. **T2.1 ‖ T2.2** — disjoint (T2.1: `hub/state.py`, `metrics/counters.py`, their tests). **T3.2 ‖ T3.3** — disjoint (neither edits the proto or `fault_proxy.py`; `metrics/hub.py` is T3.3's only, building on T2.5's `on_intake_stats`; `hub/hub.py` is T3.2's only). **T4.3 ‖ T4.4** — as part 04 says (`hub/hub.py` by function; `_count_drop` exists since T2.5, so T4.4 only adds call sites; `cli.py`, `arena/*`, `coordinator/*`, `core/run_dir.py` T4.3; `node/*`, `utils/process.py`, `core/run_state.py`, `tests/node_kit.py` T4.4). **T7.1 ‖ T7.2** — as part 07 says; T7.1 no longer touches the proto; `pyproject.toml` only T7.2. **T6.2 alongside T3.1–T5.2** — `hub/server.py`: T6.2 adds `BootstrapServicer`, `add_bootstrap` and one registration line inside `build_server`, next to T3.2's `relay=` / `latest=` keywords of the same function (adjacent hunks; the controller integrates in task order); `pyproject.toml` `known-first-party` gets `cluster_kit` / `fault_proxy` (T3.1), `eval_kit` (T5.1) and `join_kit` (T6.2) on one line: union at cherry-pick.

---

## Cross-part execution notes

- **T0.1 runs first, on the unchanged tree** (base `6fa6d8a`): its tests pass on today's code and its fixture generator `tests/fixtures/sp4/make_sp4_run.py` runs with SP4's `train`; the generator and its output are committed. T2.5 rewrites T0.1's tests against the hub with the same assertions (rotation through the deficit rule, spec 4.2).
- **Group A** (T0.1, T1.1, T1.2, T1.3, T1.4, T1.5, T6.1) runs in parallel worktrees. Shared files and how they split: `pyproject.toml` — T1.2 edits `[project].dependencies`, the extras and the ruff excludes; T1.3 adds `cryptography` and `net_kit` to `known-first-party`; `core/types.py` — only T1.2; `core/config.py` — only T1.4; `core/validation.py` — only T1.1 (`_check_state_payload`); `tests/conftest.py` — only T1.3; `tests/game_helpers.py` — only T0.1. The controller integrates group A by cherry-pick with the full fast suite after each.
- **T2.1 ‖ T2.2** after group A: T2.1 creates `hub/state.py` only; T2.2 edits `learner/learner.py`, `worker/match_runner.py`, `worker/rollout_loop.py`, `core/types.py` (`WeightPayload.generation`), `utils/seeding.py`, the two `learner_process` call sites (`launcher._learner_main`, `distributed.run_distributed_learner`, via `QueueIntake` / `QueueOutputs`) and creates `net/local.py`.
- **T2.5 switch rules** (split point allowed: (a) the hub + `train` wiring, (b) the test ports):
  - moves to `colosseum.hub.hub`: `Launcher` → `Hub` (R5 names), `run_training`, `RunSetup`, `setup_run`, `validate_run_config`, `warn_static_ownership_skew`; `_create_env` → `colosseum.node.worker_main.create_env` (and `distributed.py`'s import of it); worker / learner process bodies → `node.worker_main` / `node.learner_main`;
  - `launcher.py` is deleted; `cli.py train` calls `colosseum.hub.hub.run_training`; `scripts/bench_throughput.py::run_one` drives `Hub` (`env_steps_done`, `request_stop`) and falls back to `colosseum.launcher.Launcher` when the measured tree (`--tree`) still has it; `_make_config` adds `distributed.heartbeat_sec: 0.25`;
  - tests that poke `Launcher` internals (the 25 test files that import `colosseum.launcher`, e.g. `test_sp2_launcher_lifecycle`, `test_sp2_launcher_checkpoints`, `test_sp4_result_flow`, `test_sp3_snapshot_eviction`, `test_sp4_pulse`, `test_sp4_live_ratings`, `test_sp2_pipelines`) are ported to the `Hub` equivalents (same assertions; inbox events instead of queue items); a test is deleted only when the spec removes its subject (newest-wins mailboxes, `_resolve_lineups`'s latest fallback, `WorkerCommand` deltas per worker, `mp.Queue` release helpers); the task report lists every ported and deleted test;
  - stays until later tasks: the SP4 pulse child (spawned by the hub; removed by T5.1, R6); `distributed.py`, `transport/`, `weight_store/` and their tests (removed by T7.2; they do not use `Launcher`); `ipc.SharedCounter` / `QueueReader` while the pulse child and the old roles use them;
  - criterion 2 is measured at the end of T2.5 (bench on `train` vs `main` `d61eb02` via `--tree`, 1 / 2 / 4 workers, two runs each, alternating, by counters; the controller runs it on a quiet machine). If it fails, the task reports the profile and the controller decides.
- **Shared files after T2.5** (`hub/hub.py`, `hub/server.py`, `node/agent.py`, `cli.py`, `core/config.py`, `metrics/*`): each task edits only the handlers / sections it names in its part; new CLI commands go into new functions; when two tasks of a parallel pair touch one file, the index below names the split, and the controller integrates by cherry-pick.
- **R8 accounting** lands in T2.5 (writer commits → coordinator); its replay after a crash lands in T4.3; T4.4 adds the hub-away buffering and resend after `--resume`.
- **Generations and versions:** T2.1 has the pure registry; T2.5 uses generation 1 for every learner; T4.1 makes them real (new generation per assignment, fencing); T3.1 distributes versions across machines.
- **Pulse:** SP4 pulse child until T5.1 (R6); T5.1 moves it onto evaluator blocks and removes the child; the pulse rules (domain P, `max_backlog`, journal only, the local run's final snapshot never evaluated) are unchanged.
- **Docs** (`DISTRIBUTED_GUIDE.md`, README, CLAUDE.md, ARENA_GUIDE, LEAGUE_GUIDE) are written once in T7.5, after the measurements; earlier tasks change docs only where a test reads them.
- **Quiet-machine work** (one at a time, nothing else CPU-heavy): T1.1 and T1.5 measurements, T2.5 criterion 2, T7.4 (criteria 2–5, 9 scripts, the slow suite, the hub load test); every task's slow-suite run.
- **SP5b (R14):** part 08 runs last; if the run overruns, T8.1–T8.4 move to SP5b whole and the acceptance marks criteria 8 (SSH), 12 and 14 (K8s) «перенесено в SP5b».

## Task index and execution order

| Task | Title | Depends on | Main files (create **C** / modify **M**) | Runs |
|---|---|---|---|---|
| T0.1 | Contract tests of today's behaviour; SP4 run-dir fixture + resume test | — (unchanged tree) | C `tests/contract/test_sp5_*` baselines, `tests/fixtures/sp4/`; M `tests/game_helpers.py`, `tests/contract/test_sp4_result_flow.py` | fast |
| T1.1 | Frame codec, chunk / weights / checkpoint frames, `ChunkValidator`; criterion-5 script | — | C `net/frame.py`, `net/chunks.py`, `scripts/measure_frame_decode.py`; M `core/validation.py` | fast; measurement |
| T1.2 | Proto v1 + generated code + converters; types fields; deps move | — | C `net/proto/*`, `net/messages.py`, `scripts/gen_proto.sh`; M `core/types.py`, `pyproject.toml` (deps, extras, ruff), `scripts/setup-dev.sh` | fast |
| T1.3 | Security: identities, token, certificates, `NodeBook`, credentials, `AuthInterceptor`; dirs fixture | — | C `net/security.py`, `net/grpc_util.py`, `tests/net_kit.py`; M `tests/conftest.py`, `pyproject.toml` (cryptography, first-party) | fast |
| T1.4 | Config: `distributed`, learner keys, `transport` translation, bench config | — | M `core/config.py`, `distributed.py` (max_message_mb), `scripts/bench_throughput.py::_make_config` | fast |
| T1.5 | Shm weight cache library; criterion-4 read-latency script | — | C `node/__init__.py`, `node/weights.py`, `scripts/measure_weight_cache.py` | fast; measurement |
| T6.1 | Bundle, node requirements, torch wheel choice (pure) | — | C `bootstrap/{__init__,bundle,requirements,torch_select}.py` | fast |
| T2.1 | `hub.state`: registry, cells, epochs, rotation, budget, versions, intake, `hub_state.json` | group A | C `hub/__init__.py`, `hub/state.py` | fast |
| T2.2 | `net.local`, worker / learner IO, epoch ids and seeds, `version_floor` | group A | C `net/local.py`; M `learner/learner.py`, `worker/match_runner.py`, `worker/rollout_loop.py`, `core/types.py`, `utils/seeding.py`, `launcher.py` (`_learner_main`), `distributed.py` (learner call) | fast; slow |
| T2.3 | Hub gRPC server + node session client; blobs in parts | T1.2, T1.3, T2.1 | C `hub/server.py`, `node/session.py`, `tests/hub_kit.py`; M `net/grpc_util.py` | fast |
| T2.4 | `NodeAgent` with a fake hub: children, results buffer, lineup pull, cache fetch, heartbeats | T2.2, T2.3, T1.5 | C `node/{agent,procs,worker_main,learner_main,hardware,logs}.py`, `tests/node_kit.py` | fast |
| T2.5 | The switch: `train` = `Hub` + local node; `hub` command; R8 wiring; tests ported; criterion 2 | T2.1–T2.4 (alone) | C `hub/hub.py`, `hub/registry.py`, `hub/admin.py`; D `launcher.py`; M `cli.py`, `coordinator/coordinator.py`, `arena/matchlog.py`, `metrics/aggregator.py`, `scripts/bench_throughput.py`, `tests/cli_runner.py`, ported tests | fast; slow; gpu; measurement |
| T3.1 | Weights across machines, snapshot / fixed blobs, evicted hold, `lineup_fetch_failed`, `GetLatest` | T2.5 | C `tests/cluster_kit.py`; M `hub/hub.py` (versions, blobs), `hub/server.py` (Download), `node/agent.py` (fetch), `checkpoint_manager.py` (`.evicted`) | fast |
| T3.2 | Chunk routing, data port (mTLS), hub relay, backpressure, `dropped_chunks`; fault proxy | T3.1 | C `node/router.py`, `tests/fault_proxy.py`; M `hub/server.py` (relay servicers only), `node/agent.py` (routing section), `hub/hub.py` (collect rule) | fast |
| T3.3 | `max_policy_lag`, lag percentiles, `dropped_stale_chunks` | T3.1 | M `learner/learner.py`, `net/local.py` (`ConnectionIntake` lag rule), `metrics/hub.py` | fast; slow |
| T4.1 | Placement, devices, `pending`, Start/StopLearner with generations, fencing | T3.2, T3.3 | C `hub/placement.py`; M `hub/hub.py`, `node/agent.py`, `node/learner_main.py` | fast; slow; gpu |
| T4.2 | Recovery points, restart limits, machine loss, relocation, final blob call | T4.1 | M `learner/learner.py`, `node/procs.py`, `node/agent.py`, `hub/hub.py` | fast; slow |
| T4.3 | In-place resume, `hub_state.json` counters, R8 replay, journal `node` column + migration, SP4 dir in place | T4.2 | M `hub/hub.py` (start / resume functions only), `arena/matchlog.py`, `coordinator/coordinator.py`, `core/run_dir.py`, `cli.py` (`--resume`) | fast |
| T4.4 | Hub away, buffering + resend, lifecycle and stop, exit codes, log tails, portability shims | T4.2 | M `node/agent.py`, `node/session.py`, `node/procs.py`, `node/logs.py`, `hub/hub.py` (stop / exit-code functions only), `utils/process.py`, `core/run_state.py` | fast |
| T5.1 | Evaluators, block leases, pulse on blocks; pulse child removed | T4.3, T4.4 | C `hub/evaluation.py`, `node/evaluator_main.py`; M `arena/pulse.py`, `hub/hub.py`, `node/agent.py` | fast |
| T5.2 | Hub-run tournaments, `tournament --hub`, `hub --idle` | T5.1 | M `hub/evaluation.py`, `hub/admin.py`, `arena/tournament/run.py`, `cli.py` (tournament, hub --idle) | fast |
| T6.2 | Bootstrap service, installer, loader (fake uv, no network) | T2.5, T6.1 (parallel with T3.x–T5.x) | C `bootstrap/installer.py`, `bootstrap/loader.py`; M `hub/server.py` (`BootstrapServicer` class and its registration line only) | fast |
| T6.3 | `join`, persistent nodes, registration checks, `nodes list/remove`, `hub token`, join commands, WSL2 message | T6.2, T5.2 | M `cli.py` (join, nodes, hub token), `hub/hub.py` (registration), `hub/admin.py` (nodes commands), `node/agent.py`, `tests/cli_runner.py` | fast |
| T7.1 | Metrics keys, `nodes.json`, nodes line, «Nodes» section, `hub --dashboard` | T6.3 | M `metrics/*`, `arena/report/*`, `hub/hub.py` (metrics tick), `cli.py` (--dashboard) | fast |
| T7.2 | Old roles removed (exit 2), old modules and tests removed, `validate` placement + `distributed:` line, logging before checks | T6.3 (parallel with T7.1) | D `distributed.py`, `transport/`, `weight_store/`, `proto/colosseum.proto`, old tests; M `cli.py` (old commands, validate), `core/validation.py`, `pyproject.toml` (ruff excludes) | fast |
| T7.3 | Node image, compose, `-m docker` tests; old deployment removed | T7.1, T7.2 | C `deployment/Dockerfile.node`, `compose*.yaml`; D old `deployment/*`; M `pyproject.toml` (marker) | fast (docker skipped without Docker) |
| T7.4 | Measurements (criteria 2–5, 9) and slow tests (localhost pipeline, hub load) | T7.3 | C `scripts/measure_hub_load.py`, `scripts/measure_scaling.py`, slow tests; M `docs/benchmarks.md` | slow; measurement |
| T7.5 | Docs: `DISTRIBUTED_GUIDE.md`, README, CLAUDE.md, ARENA_GUIDE, LEAGUE_GUIDE | T7.4 | docs only | fast |
| T7.6 | Acceptance runbook (exact commands, thresholds, owner actions per criterion; no code) | T7.5 | part 07 text | — |
| T8.1 | SSH mode (`nodes add`, reverse tunnel, remembered nodes, `--nodes`) | T7.6 | M `hub/hub.py`, `hub/admin.py`, `cli.py`; fake `ssh` tests | fast |
| T8.2 | Agents on the fly, dynamic membership, `on_agents_changed`, `agents_state.json` | T8.1 | M `coordinator/*`, `league/base.py`, `league/pfsp.py`, `metrics/hub.py`, `arena/live.py`, `hub/hub.py`, `cli.py` | fast |
| T8.3 | `nodes update-code` (rolling restarts, refusals, learner restarts on changed architecture code) | T8.2 | M `hub/hub.py`, `node/agent.py`, `bootstrap/bundle.py`, `cli.py` | fast |
| T8.4 | K8s manifests + static validation | T8.3 | C `deployment/k8s/*`; structural test | fast |

**Execution order:** group A in parallel → T2.1 ‖ T2.2 → T2.3 → T2.4 → T2.5 (alone) → T3.1 → T3.2 ‖ T3.3 → T4.1 → T4.2 → T4.3 ‖ T4.4 → T5.1 → T5.2 → T6.3 → T7.1 ‖ T7.2 → T7.3 → T7.4 → T7.5 → T7.6 → T8.1 → T8.2 → T8.3 → T8.4. T6.2 may run in parallel with any task from T3.1 to T5.2 (it only adds `BootstrapServicer` to `hub/server.py` and the `bootstrap/` modules). Parallel pairs and their split of shared files: T3.2 ‖ T3.3 (T3.2: router, relay servicers, the hub's collect rule; T3.3: learner and `ConnectionIntake` drop rules, `metrics/hub.py`); T4.3 ‖ T4.4 (`hub/hub.py`: T4.3 the start / resume functions `_resume_*`, `_load_hub_state`, `_replay_journal`; T4.4 `_shutdown`, `_stop_nodes`, `_on_hub_lost`-side counters and exit-code functions; `cli.py`: only T4.3); T7.1 ‖ T7.2 (`cli.py`: T7.1 `hub --dashboard` only, T7.2 the old commands and `validate`).

After T7.6: final whole-branch review (three areas), one fix wave and a scoped re-review, the one-machine acceptance (criteria 1–7, 13, 14 where possible), then the two-machine acceptance with the owner (criteria 8–12, 15) — controller tasks following T7.6's runbook.

## Traceability (spec item → task)

| Spec | Item | Task(s) |
|---|---|---|
| 1 | hub process, run-dir ownership, all-interface listen, `--listen` | T2.5 |
| 1 | node agent; children die with it | T2.4, T4.4 |
| 1 | learner on any node, device-based placement | T4.1 |
| 1 | `train` = hub + local child node; loopback ephemeral port, `hub.json`, running registry, `--listen`; persistent / SSH nodes only for `hub` and `train --listen` | T2.5, T6.3, T8.1 |
| 1 | `hub --idle` | T5.2 |
| 2.1 | hub identity per machine, token, node certificates, `--token-file` / `COLOSSEUM_TOKEN` | T1.3, T6.3 |
| 2.2 | join command, one-liner, SSH mode, `nodes list` / `remove [--ban]` | T6.3, T6.2, T8.1 |
| 2.3 | thin loader, bundle by hash, per-process `PYTHONPATH`, project cwd | T6.1, T6.2 |
| 2.4 | requirements freeze, Python major.minor, torch build by hardware, uv envs | T6.1, T6.2 |
| 2.5 | registration checks | T6.3 |
| 2.6 | hardware report, worker / evaluator / learner slots, `pending` | T2.4, T4.1 |
| 3.1 | ports, data port, local relay, port check | T1.4, T2.3, T3.2 |
| 3.2 | TLS (R1), fixed server names | T1.3 |
| 3.3 | `--insecure`, loopback, admin target resolution | T1.3, T2.5 |
| 3.4 | no pickle, caps before allocation, result checks, `weights_only` | T1.1, T2.1, T4.2 |
| 3.5 | WSL2 NAT detection + hint | T6.3 |
| 3.6 | `cryptography`, uv | T1.3, T6.2 |
| 4.1 | session messages, resume, node loss | T1.2, T2.3, T2.4 |
| 4.2 | cells, epochs, TTL pull, lineup after cache, no latest fallback, evicted hold, deficit rotation, no idle cells, hub-side matchmaker | T2.1, T2.4, T2.5, T3.1 |
| 4.3 | result batches, buffer, ack after journal, uids, journal `node` column, intake checks, hub-stamped `env_steps` | T2.1, T2.4, T2.5, T4.3 |
| 4.4 | one budget, `hub_state.json`, LR progress from the hub, WARNING at / past budget | T2.1, T2.5, T4.3 |
| 4.5 | version registry, monotonic versions, `GetLatest` without the hub, `hub_lost_collect_sec` | T2.1, T3.1, T4.1, T4.4 |
| 4.6 | shm cache, one fetch per machine, `teacher_active`, fixed / teacher / init by id, space checks | T1.5, T2.4, T3.1 |
| 4.7 | per-process connections, one learner intake path, backpressure, at-most-once with reasons, no collecting seats without a learner, `max_policy_lag`, lag metrics | T2.2, T3.2, T3.3, T4.1 |
| 4.8 | relay via the hub, `relay_bytes`, `relay: false` | T3.2 |
| 4.9 | frame format, hostile input, chunk checks, adaptive lz4, typed protobuf, protocol version | T1.1, T1.2 |
| 4.10 | old roles / modules removed, `transport.*` translation, logging before checks | T1.4, T7.2 |
| 5.1 | SP3/SP4 logic in the hub unchanged; journal watermark in `league_state.json` | T2.5, T4.3 |
| 5.2 | snapshots via the hub, recovery points, trainer state as bytes | T2.5, T4.2 |
| 5.3 | evaluators, leases, pulse on blocks, `tournament --hub`, `hub --idle` | T5.1, T5.2 |
| 5.4 | in-place `--resume`, refusal while live, config precedence, SP4 dir in place, `resume_from` unchanged | T4.3 |
| 6.1–6.3 | agent set, `agents …`, dynamic membership, `on_agents_changed`, schedules from the start point | T8.2 |
| 6.4 | `nodes update-code` | T8.3 |
| 7.1 | failure table (worker / evaluator restarts, contract errors, machine loss, learner crash, relocation, network cut, hub away, bad data, full shm) | T4.2, T4.4, T3.2 |
| 7.2 | generations | T2.1, T4.1 |
| 7.3 | not automated (recipes) | T7.5 |
| 7.4 | stop, grace, final snapshot by a dedicated call | T4.2, T4.4 |
| 7.5 | exit codes | T2.5, T4.4, T6.3, T5.2 |
| 7.6 | monotonic clocks, cell seeds | T2.1, T2.2 |
| 7.7 | node logs, tails in the run dir | T2.4, T4.4 |
| 7.8 | portability (parent death, liveness, shm dir, socket paths) | T1.5, T2.2, T4.4 |
| 8 | metrics, `nodes.json`, «Nodes» section, `hub --dashboard`, console | T3.3, T4.1, T7.1 |
| 9.1 | config keys | T1.4 |
| 9.2 | commands | see the CLI table |
| 10 | Docker image + compose; K8s manifests | T7.3; T8.4 |
| 11 | documentation | T7.5 |
| §6 | contract tests before the move; frame, hub-with-fakes, node-with-fake-hub, localhost multi-node, security, environment, config, slow, docker tests | T0.1, every task's tests, T7.4 |
| §3 | criteria 1–7 (one machine), 8–15 (two machines) | T7.4 (measurements, slow tests), T7.6 (runbook) |

---

## Questions for the controller

- **Q1 — the hub's TLS certificate is the CA certificate.** Spec 2.1 has the CA issue a separate hub server certificate. The node must obtain the CA certificate before it trusts the hub; Python 3.12's `ssl` returns only the leaf of an unverified handshake, and grpc-python has no "fetch the chain" API. Proposed resolution (used in the contract): the self-signed CA certificate (CA:TRUE, SAN `colosseum-hub`, EKU serverAuth) is also the hub's TLS certificate; the node fetches it with `ssl.get_server_certificate`, checks its SPKI fingerprint against the token, and uses it as `root_certificates`; node certificates are issued by it for data-port mTLS. T1.3 verifies the handshake against a real loopback gRPC server first; if BoringSSL rejects a CA certificate as a server leaf, the fallback is a separate leaf that carries the CA certificate DER in a private X.509 extension (the node checks the embedded CA's fingerprint and the leaf's signature with `cryptography`).
- **Q2 — stale-generation fencing by version cap.** Spec 7.2 rejects results and chunks of "a stale generation". Read literally, every learner restart would drop the in-flight results and chunks of legitimately announced versions of the previous generation. Proposed resolution (used in the contract): reject a seat result / chunk of generation `g` only when `g` is superseded and its version is above `cap(g)` (the highest version the hub accepted from `g` before assigning `g + 1`); versions ≤ `cap(g)` are announced, unique and kept.
- **Q3 — R4 versus epochs.** R4 says `RolloutLoop` stays unchanged behind a new `LoopIO`, but spec 4.3 / 7.6 put the cell epoch into every training match id and episode seed, which `LoopIO` cannot reach (the runner forms both). Proposed resolution (used in the contract): `Lineup.epoch` (T1.2) plus the optional keywords `cell_seed_fn` (`MatchRunner`, `RolloutLoop`) and `uid_prefix` (`RolloutLoop`, for the `n<node>/` uid segment) (T2.2); without them both behave exactly as in SP4, so the SP1–SP4 contract tests of the loops stay valid.
- **Q4 — slot counts of `train`'s local node.** Spec 2.6 gives one learner slot per GPU and one CPU learner on ≥ 4 cores; applied to the local node of `train`, a two-agent config on one GPU (e.g. `tic_tac_toe_multi.yaml`) would leave an agent `pending`, breaking criterion 6 ("configs train on one machine without edits"). Proposed resolution (used in the contract): the local node (`NodeOptions.local`) takes every trainable agent's learner (a GPU takes any number, CPU learners unlimited), evaluators = 1 iff `evaluation.enabled`; spec 2.6 defaults apply to `join` nodes.
