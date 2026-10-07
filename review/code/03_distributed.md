# R3: Distributed layer, learner, launcher, CLI (adversarial review)

Scope: `src/colosseum/distributed.py`, `launcher.py`, `cli.py`, `learner/learner.py`, `weight_store/*`, `transport/*`, `proto/colosseum.proto`, the tests that cover them, and `deployment/`.
Environment: 8-core WSL box shared with other agents. Load average was 20–35 during the runs, so absolute latencies are inflated (roughly 2–5x). Ratios are still meaningful.
Artifacts: scripts are in `scratchpad/r3_dist/scripts/` (`bench_serialization.py`, `probes.py`, `probe_maxmsg.py`, `scenario.sh`) and logs are in `scratchpad/r3_dist/logs/`.

---

## Actual deployment capabilities (what really works today)

### Processes and CLI commands that exist

| Command | What it runs | Needs |
|---|---|---|
| `colosseum train -c cfg.yaml` | **Single machine only.** Launcher, coordinator (matchmaking, checkpoints, ELO) in the main process, plus 1 learner process per trainable agent and `rollout.num_workers` worker processes. Everything is connected with `mp.Queue`. | config |
| `colosseum serve-weight-store --port P` | gRPC WeightStore (in-memory, latest weights per agent_id). | — |
| `colosseum run-learner -c cfg --agent A --traj-port P --weight-store H:P` | One agent's learner. It runs an in-process gRPC TrajectoryService on `P`, trains, pushes weights to the store, and writes checkpoints to `checkpoint.dir` on the learner's local disk. | config file, weight store address, a unique port per agent |
| `colosseum run-workers -c cfg --weight-store H:P -l A=host:port [-l B=host:port ...]` | `rollout.num_workers` worker processes on this host. They send chunks to each agent's learner and poll the store for weights. | config file (must match the learners'), every learner address |
| `colosseum serve-trajectory --port P` | A TrajectoryService whose queue **nobody reads**. It is a dead end: after 256 chunks every RPC blocks for 5 s and then drops the chunk. | — |

There is no discovery. Every address and port is passed by hand. `transport.mode` and `transport.grpc_port` are **never read** by any code (grep shows they are only used in `core/config.py`).

### Topologies that work (VERIFIED-BY-RUNNING on localhost, separate processes via the CLI)
- **Machine A: weight store. Machine D (GPU): `run-learner` for agent X and a second `run-learner` for agent Y on another port. Machines B, C: `run-workers -l X=D:p1 -l Y=D:p2`.**
  - Verified with 2 learners plus 1 worker fleet using `tic_tac_toe_multi.yaml`: both learners trained (`step=400`) and both wrote checkpoints.
  - Learners for different agents can sit on different machines, because each one is an independent process with its own port.
- **A second worker machine can join mid-training.** In `scenario.sh`, starting a second `run-workers` raised the learner rate from about 11 to about 30 train steps/s.
- **A worker machine can leave.** Its chunks simply stop arriving. However, `SIGTERM` to the `run-workers` parent **orphans its worker children**, which keep running (R3-17).
- Two of the paths in this list (the gRPC flow and the K8s/compose manifests) assume the `run-learner`/`run-workers` commands. **The README's documented flow (`serve-trajectory` plus `train --set transport.mode=grpc`) does not distribute anything** (R3-01).

### What does NOT work in distributed mode
- **No coordinator.** No matchmaking, PFSP, historical-checkpoint opponents, frozen or scripted opponents, match results, ELO or win rates (R3-02). The workers use a static round-robin `slot_agent_map`. With 1 agent, every match is latest-vs-latest self-play. With 2 agents, alpha always plays beta (never itself). The `distributed.py:18-22` docstring admits this.
- **Remote workers never load historical checkpoints.** Checkpoints exist only on the learner's local disk (or on an RWO PVC in K8s). Nothing serves them.
- **No new agent can be added mid-training.** The agent list is fixed by each worker host's `-l` flags. Adding an agent means restarting every worker fleet. The same is true in local mode (agents are fixed at launch).
- **No elasticity on the control plane.** If the weight store restarts, the learner and every worker crash (R3-03). If the learner restarts, it begins again at version 0 with no resume (R3-05), workers ignore its weights forever (R3-04), and its fresh checkpoints are deleted (R3-06).
- **No metrics.** No WandB in distributed mode (`metrics_queue=None`). Worker-process logs are lost (R3-15, R3-16).
- **Run length is not coordinated.**
  - The learner stops after `total_timesteps/(T*batch_chunks)` train steps.
  - Each `run-workers` host independently runs the **full** `total_timesteps`.
  - After the learner exits, workers keep generating and silently dropping chunks (34 s of wasted rollouts in `run_distributed_test`) (R3-18).

### Simplicity: steps for a 3-machine setup today
For "A = weight store, B = GPU learner(s), C = CPU workers":
1. Copy the same YAML and the user's env/network package to all 3 machines (no config distribution, no consistency check).
2. A: `colosseum serve-weight-store --port 50051`.
3. B: `colosseum run-learner -c cfg --agent agent_0 --traj-port 50052 --weight-store A:50051`. Repeat with a new port per agent.
4. Start the store **before** the learner, or the learner crashes (R3-03).
5. C: `colosseum run-workers -c cfg --weight-store A:50051 -l agent_0=B:50052 [...]`, with `--set rollout.num_workers=<cores>` per host.
6. Watch per-process stdout. There is no aggregated metrics view.

That is 3+ commands, N hand-assigned ports and N `-l` flags per worker host. The result is still only latest-vs-latest self-play.

---

## Findings

### R3-01: high, doc-mismatch/ux. The README's distributed instructions run N independent single-machine trainings
- **Location:** `README.md:416-436`, `cli.py:218-237`, `core/config.py:245-252`.
- **What is wrong:**
  - The README says "Machine 2: `colosseum serve-trajectory`; Machine 3+: `colosseum train --set transport.mode=grpc`".
  - `transport.mode` is never read, so `train` always builds the local `mp.Queue` pipeline. Each "worker machine" therefore runs its own isolated learner, workers and coordinator.
  - `serve-trajectory` creates a `queue.Queue(maxsize=256)` that nothing consumes.
  - The real commands (`run-learner`, `run-workers`) are not documented in the README. CLAUDE.md still lists "Distributed launcher that uses gRPC transport" as an open item and says the CLI has only `serve-*`.
  - `deployment/Dockerfile.worker` still has `ENTRYPOINT colosseum train`.
- **Evidence:** BY-READING (grep shows no use of `transport.mode` or `grpc_port` outside config.py). The `serve-trajectory` behaviour (block for 5 s, then drop) was VERIFIED-BY-RUNNING via the same servicer in `probes.py backpressure`: sends 4–6 took 5.01 s each with `chunks_received=0`.
- **Impact:** A user who follows the docs gets no distribution, and nothing tells them so.
- **Fix:**
  - Delete `serve-trajectory` and `transport.mode`/`grpc_port`, or make `train` fail loudly if `mode=grpc`.
  - Document `run-learner`/`run-workers`, update CLAUDE.md, and drop `Dockerfile.worker`.

### R3-02: critical, design. Distributed mode has no coordinator: no league, no PFSP, no historical opponents, no results
- **Location:** `distributed.py:18-22, 246-304, 331-337`.
- **What is wrong:**
  - `_dist_worker_target` passes no `results_queue`, `command_queue`, `slot_network_map`, `collect_mask` or checkpoint dicts.
  - `slot_agent_map` is a fixed round-robin `agent_ids[(e*num_players+p) % len(agent_ids)]`.
  - Every slot collects data, there are no frozen or scripted agents, and match outcomes are discarded, so no ELO or win rates are computed.
  - Checkpoints are written by each learner to its local `checkpoint.dir`, and nothing serves them to workers.
- **Evidence:** BY-READING, confirmed by the module docstring. In the 2-agent distributed run, both learners received exactly the same number of chunks (alpha vs beta in every env).
- **Impact:** The owner's #1 priority (rollouts on CPU boxes, training on GPU boxes) works only for the simplest phase. Self-play vs history (Phase 3) and PFSP/league (Phase 4) are single-machine only. Latest-vs-latest self-play is prone to strategy cycling.
- **Fix:** Turn the coordinator into a gRPC service ("hub", see the target design):
  - `GetAssignment(worker_id)` returns a `WorkerCommand`-equivalent: slot maps plus checkpoint ids.
  - `ReportResults`.
  - `GetCheckpoint(agent, ckpt_id)` streaming.
  - The worker loop already supports `command_queue` and `results_queue`; give them gRPC-backed adapters the same way `GRPCWeightSource` adapts the weight queue.

### R3-03: high, bug (fault tolerance). Any weight-store blip kills the learner and every worker
- **Location:**
  - `distributed.py:92-100` (`GRPCWeightSource.get_nowait` calls `get_version` with no `RpcError` handling).
  - `distributed.py:110-111` (`GRPCWeightSink.put_nowait`).
  - `learner.py:209-215` (`_push_weights` only catches `Full`).
  - `grpc_store.py:166-169`.
- **Evidence:** VERIFIED-BY-RUNNING:
  - Starting `run-learner` before the store led to `exit=1` with `_InactiveRpcError UNAVAILABLE` raised from `learner.py:85 _push_weights → distributed.py:111 → grpc_store.py:119`.
  - In `scenario.sh`, killing and restarting the store gave: "learner alive? no" and "workers A alive? no". The worker traceback runs `rollout_worker.py:596 → _sync_weights:731 → distributed.py:93 get_nowait → grpc_store.py get_version`. `run-workers` then logged "Distributed workers stopped." and exited 0.
  - `probes.py weight_store_down`: `get_nowait raised _InactiveRpcError StatusCode.UNAVAILABLE`.
- **Impact:**
  - Start order is mandatory.
  - A store pod reschedule in K8s takes down the whole run.
  - Because the store is in-memory, a restart also loses all weights. Workers then keep their old weights until the learner pushes again (fine), but they never get that far because they crash.
- **Fix:**
  - Catch `grpc.RpcError` in the source and sink: Empty or skip-and-retry with backoff.
  - Use `wait_for_ready=True` plus deadlines on the RPCs.
  - Have the learner keep its latest payload and re-push it when the store returns `version=-1`.

### R3-04: high, bug. A learner restart makes workers ignore its weights indefinitely (version regression)
- **Location:** `distributed.py:90-100` (`if version <= self._last_version: raise queue.Empty`), together with versions restarting at 0 (R3-05).
- **Evidence:** VERIFIED-BY-RUNNING (`probes.py version_regression`): "worker got v 500 … after learner restart (store at v10) worker sees queue.Empty -> keeps stale v500 weights".
- **Impact:**
  - The K8s learner Deployment restarts the container whenever it exits, including after a normal finish or an OOM.
  - Every worker then keeps acting with the pre-crash policy until the new learner overtakes the old version number. Meanwhile it labels chunks with the stale `behavior_policy_version`.
  - This is silent.
- **Fix:**
  - Version = (incarnation/run_id, step). Workers reload when the incarnation changes.
  - Or let the store return a monotonically increasing store-side sequence number.
  - Or reject `put` with a lower version unless the request carries a new incarnation.

### R3-05: high, bug/fault-tolerance. `run-learner` cannot resume
- **Location:** `distributed.py:221-233`. `learner_process` is called without `resume_state`, so `training.resume_from` is ignored. Compare `launcher.py:402` (`_resolve_resume_state`, local mode only).
- **Evidence:** BY-READING.
- **Impact:** Any distributed learner crash or restart starts training from random init. This combines with R3-04 and R3-06.
- **Fix:** Call `_resolve_resume_state` with a `CheckpointManager` built on `checkpoint.dir`, and default to "resume from the latest checkpoint if one exists" (`--resume auto`).

### R3-06: high, bug. After a restart without resume, newly saved checkpoints are deleted
- **Location:** `coordinator/checkpoint_manager.py:111-121`.
- **What is wrong:**
  - `_scan_existing` indexes the old `ckpt_v20/40/60`.
  - The restarted learner saves `ckpt_v20` again (same directory), and the code appends a duplicate index entry.
  - FIFO eviction then pops the oldest entry, which is the old `ckpt_v20` with the same path, and `rmtree`s the directory it just wrote.
- **Evidence:** VERIFIED-BY-RUNNING (pool_size=3):
  - `index: ['ckpt_v40', 'ckpt_v60', 'ckpt_v20']`
  - `on disk: ['ckpt_v40', 'ckpt_v60']`
  - `ckpt_v20/model.pt exists: False`
- **Impact:** Fresh checkpoints vanish. The index holds dangling entries, so later loads raise `FileNotFoundError` and the launcher falls back to "latest". Old-run checkpoints get mixed into the new run's opponent pool.
- **Fix:**
  - De-duplicate by id on save (replace the entry, do not append).
  - Make the checkpoint id include the run incarnation.
  - Write atomically: temp dir, then `os.replace`.

### R3-07: high, bug. Local mode: the final checkpoint is lost and `colosseum train` crashes at the end of the run
- **Location:** `launcher.py:520-556` and `learner.py:121-138`. Root cause: torch shares CPU tensors through `mp.Queue` via fd passing (`resource_sharer`), so the producer must still be alive when the consumer unpickles.
- **Evidence:** VERIFIED-BY-RUNNING, 3 of 3 runs. Command: `train --set training.total_timesteps=1600 rollout.chunk_length=8 learner.batch_chunks=2 self_play.checkpoint_interval=50` (100 train steps, so the expected checkpoints are v50 and v100).
  - Result each time: `exit=1`, only `ckpt_v50` on disk.
  - Traceback: `launcher.py:546 cq.get_nowait() → torch/multiprocessing/reductions.py rebuild_storage → resource_sharer … FileNotFoundError`.
  - The learner put the v100 snapshot and exited immediately, before the main process unpickled it.
  - When the race goes the other way, `_monitor_loop` sees every learner dead and `break`s **before** draining the checkpoint queues (line 522-526), so the checkpoint is lost silently.
- **Impact:** The final trained model is never saved in local mode unless `total_train_steps % checkpoint_interval == 0` *and* the race is won. Even then the process exits 1.
- **Fix:**
  - Have the learner save its own final checkpoint before exiting.
  - Drain the queues before the liveness check.
  - Catch `(OSError, EOFError)` on `get`.
  - Better: avoid tensor fd-sharing for messages that outlive the producer. Send bytes (`torch.save` into a buffer, or raw numpy), or set the `file_system` strategy.

### R3-08: high, bug (fault tolerance). Local mode: one worker dying kills the learner and silently ends the run
- **Location:** `learner.py:179-191` (`_collect_chunks` catches only `Empty`). Same fd-sharing mechanism as R3-07.
- **Evidence:** VERIFIED-BY-RUNNING. `train` with 2 workers, then `kill -9` on one worker after 25 s:
  - Learner `Process-1` raised `ConnectionResetError: [Errno 104]` inside `learner.py:188 _collect_chunks`. It was unpickling a chunk from the dead worker.
  - The monitor then logged "All learner processes exited, stopping workers…" and the shell reported `Done` (exit 0).
  - The same mechanism makes workers crash in `_sync_weights` when they read a weight payload after the learner exits (`run_scaled_test`: `Process-4/5 FileNotFoundError`).
- **Impact:** A single env bug or OOM in one rollout process ends the whole training run, and the exit code says success.
- **Fix:**
  - Catch unpickle errors per item and skip the chunk.
  - Serialize chunks to bytes, or use a shared-memory ring buffer the learner owns.
  - Restart dead workers from the monitor loop.
  - Return a nonzero exit code when a learner dies with a nonzero exitcode.

### R3-09: high, bug. Multi-agent local mode: if one learner crashes, the whole run hangs silently
- **Location:**
  - `rollout_worker.py:539-545`: `while not stop_event.is_set(): tq.put(chunk, timeout=1.0)` blocks the worker forever once the dead agent's queue is full.
  - `launcher.py:522-526`: the run stops only when **all** learners have exited.
- **Evidence:** VERIFIED-BY-RUNNING (`multi_crash.yaml`, agent_beta given a bad `algorithm_class`):
  - Beta's learner died with `ModuleNotFoundError`.
  - After 70 s, agent_alpha had not reached 10 train steps (no `step=` line).
  - The main process and processes 9712/9714 were still alive at 0–4 % CPU.
- **Impact:** A league run with N agents stalls forever on one bad learner, and there is no error in the main log.
- **Fix:**
  - Treat any learner exit with a nonzero `exitcode` as fatal, or restart that learner.
  - Make worker puts to a dead agent's queue non-blocking with drop.
  - Run `validate_config` on the algorithm (import `algorithm_class`) before spawning.

### R3-10: high, correctness/performance. Local mode weight sync drops the newest weights, so workers run policies about one sync interval stale
- **Location:** `learner.py:209-215` (`put_nowait` into `maxsize=2` queues, skip on `Full`) together with `rollout_worker.py:726-738` (drain and take the last item).
- **What is wrong:** The two queued items are the first two pushes after the previous drain. Every later push is discarded.
- **Evidence:** VERIFIED-BY-RUNNING (`probes.py local_weight_queue_staleness`): the learner pushed v1..v40 between two syncs, and "worker _sync_weights loaded v2".
- **Impact:** Workers act with weights about `weight_sync_interval_sec` old (2–5 s by default) instead of fresh ones. A fast learner can be 100+ versions ahead. That increases V-trace truncation and slows learning. The distributed path does not have this bug, because it reads the latest version from the store.
- **Fix:**
  - Use a single-slot "latest" mailbox per worker: drain-then-put on the learner side, or a shared-memory buffer plus a version counter.
  - At minimum, `get_nowait` the stale item before `put_nowait`.

### R3-11: high, correctness. The learner trains on whatever is in the queue (often 1 chunk), not on `batch_chunks`
- **Location:** `learner.py:166-193`. `_collect_chunks` waits only for the first chunk, then takes items non-blocking until the queue is empty.
- **Evidence:** VERIFIED-BY-RUNNING:
  - `scenario.sh` with `batch_chunks=4`: `step=330, chunks=332` and `step=1080, chunks=1180`.
  - 2-agent distributed run with `batch_chunks=2`: `step=400, chunks=400`.
- **Impact:**
  - When the learner is faster than the workers, which is the expected setup on a GPU box, the effective batch is 1 chunk (T samples).
  - The result is high-variance gradients, and advantage normalization runs over a single chunk.
  - The LR schedule and `total_train_steps` count steps, so training stops after consuming only about 1/`batch_chunks` of the planned env steps.
  - GPU utilization stays tiny.
- **Fix:** Block until `batch_chunks` chunks have arrived (with a stop-aware timeout), optionally with a `min_batch_chunks`. Count progress in consumed env steps, not train steps.

### R3-12: high, performance/bug. The weight plane: 64 MiB cap, lz4 that does nothing, full re-serialization for every pull
- **Location:** `grpc_store.py:63-91`, `serialization.py:12-27`, `config.py:250` (`grpc_max_message_mb=64`).
- **What is wrong:**
  1. Weights are sent as one protobuf message. With the default 64 MiB cap, any model with more than about 16.7M fp32 params fails. Protobuf also has a hard 2 GiB limit.
  2. lz4 on float weights gives a compression ratio of **1.000**: pure CPU cost.
  3. The server stores weights by `torch.load` + `torch.save` on Put. On **every** `GetWeights` it runs `torch.load` + `torch.save` + lz4 again, under the GIL, on a 4-thread pool.
  4. Every worker *process* (not every machine) pulls the full weights.
  5. The learner's push is synchronous inside the training loop.
- **Evidence:** VERIFIED-BY-RUNNING:
  - `probe_maxmsg.py` (18.4M params): `PutWeights failed: RESOURCE_EXHAUSTED Sent message larger than max (73733642 vs. 67108864)`. In a real learner this is an uncaught exception at `_push_weights`, so the learner crashes (same path as R3-03).
  - Benchmark (loaded machine): see the measured numbers below.
- **Impact:**
  - Medium-size models (e.g. Lux-style CNNs above 17M params) cannot run distributed at all with the default settings.
  - With 50 worker processes × 2 agents pulling every 5 s and a 4M-param model, the store needs about 20 req/s × 125 ms, roughly 2.5 CPU-seconds per second. That is more than a GIL-bound server can deliver.
  - Network traffic is about 160 MB/s, all of it redundant within each machine.
  - Each learner push stalls training for about 100 ms or more.
- **Fix:**
  - Store the serialized bytes once per version and return them as-is.
  - Server-stream the weights in 1–4 MiB pieces.
  - Drop lz4 for weights; optionally send them as bf16/fp16.
  - Use one pull per machine, shared with local processes through shared memory.
  - Push from a background thread.

### R3-13: high, design/performance. Trajectory backpressure: the worker blocks 5 s, then the chunk is silently dropped. One stream per chunk, sent synchronously in the env loop
- **Location:** `grpc_transport.py:36-50` (`self._queue.put(chunk, timeout=5.0)`, then a warning and the drop), `grpc_transport.py:103-112` (`SendChunks(iter([proto]))`, a new stream for every chunk), `distributed.py:62-73` (drops at `logger.debug`).
- **Evidence:** VERIFIED-BY-RUNNING:
  - Backpressure probe: once the queue was full, each send took **5.01 s** and returned `chunks_received=0`. The client ignores that count.
  - `run_distributed_test`: "Trajectory queue full, dropping chunk".
  - Stream-per-chunk vs one stream for many chunks (localhost, loaded): 21.8 ms vs 3.4 ms per chunk (T=32), and 27.9 ms vs 12.0 ms (T=256, obs 256).
- **Impact:**
  - When the learner is the bottleneck, workers stall *and* data is wasted. This is neither clean backpressure nor cheap drop.
  - No metric records how much is dropped.
  - Each send blocks env stepping for milliseconds even when the learner is idle. Example: 16 chunk sends × about 20 ms = 0.3 s per 256-step rollout per worker process.
- **Fix:**
  - One persistent client stream per (worker process, learner) fed by a background sender thread.
  - A bounded local outbox with a drop-oldest policy, plus counters.
  - Server-side flow control: do not block for 5 s. Either reject fast with a "slow down" signal, or let gRPC's HTTP/2 flow control apply backpressure by not reading the stream.
  - Expose `chunks_dropped`, `queue_depth` and `send_latency`.

### R3-14: medium, performance. `torch.save` + lz4 chunk serialization is 10–25x slower than raw buffers
- **Location:** `serialization.py:30-83`.
- **Evidence:** VERIFIED-BY-RUNNING (`bench_serialization.py`, single thread, loaded machine):

| Chunk | Current size | Current ser / deser | Raw numpy size | Raw ser / deser |
|---|---|---|---|---|
| tictactoe T=32 | 2.5 KiB | 338 µs / 793 µs | 4.1 KiB (lz4 1.3 KiB) | 12 µs / 77 µs |
| CLAUDE.md "typical" T=256, obs 256, dense float | 261 KiB (lz4 saves nothing) | 592 µs / 930 µs | 262 KiB | 27 µs / 85 µs |
| same, sparse binary obs | 51.5 KiB | 765 µs / 1091 µs | lz4: 50 KiB | 311 µs / 85 µs |
| image-ish T=128, 8×32×32 | 746 KiB | **27.3 ms / 22.4 ms** | lz4: 744 KiB | 25 ms (lz4) / 0.08 ms |

- **Impact:** About 1–2 ms of GIL-held CPU per chunk on both ends. On the learner, this deserialization runs in gRPC threads that compete with the training loop for the GIL. lz4 helps only for sparse/binary observations, and it costs about 25 ms per large chunk.
- **Fix:**
  - Fixed header (dtype, shape, offset per field) plus raw contiguous buffers, decoded with `np.frombuffer` and `torch.from_numpy` (zero-copy).
  - lz4 only on the observations field, and only when it is configured or the measured ratio is below 0.7.
  - This also removes the `torch.load` attack surface (R3-22).

### R3-15: medium, ux/observability. No metrics in distributed mode, and no throughput, lag or queue metrics anywhere
- **Location:**
  - `distributed.py:229` (`metrics_queue=None`).
  - `learner.py:141-155`: only algorithm losses, `train_step` and `chunks_received`.
  - `behavior_policy_version` is never read by the learner or algorithm (grep).
- **Evidence:** BY-READING. Distributed learners print one log line every 10 steps. Nothing goes to WandB.
- **Impact:** The owner cannot tell whether the GPU box is starved or saturated, how stale workers' policies are, or how many chunks are dropped. These are exactly the numbers needed to decide how many CPU boxes to add.
- **Fix:** Per learner, emit:
  - env-steps/s consumed
  - samples per batch
  - queue depth
  - policy lag (`current_version - behavior_policy_version`: mean and max)
  - time split between waiting for data, H2D, compute and weight push

  Per worker, emit:
  - env steps/s
  - chunks sent and dropped
  - send latency
  - weight version age

  Let every process log to WandB with the same run id or group, or send stats to the hub.

### R3-16: medium, ux. Child-process logs are lost in `colosseum train` and `run-workers`
- **Location:** `launcher.py:58-124, 127-193` and `distributed.py:246-304`. Spawned children never call `logging.basicConfig`, so INFO-level logs are dropped and only warnings reach stderr through the last-resort handler.
- **Evidence:** VERIFIED-BY-RUNNING:
  - `grep -c learner.learner logs/train_final_ckpt.log logs/train_multi.log` returned 0 and 0.
  - `sc_wA.log` contains no "Worker 0: starting/finished" lines.
  - `run_distributed_test.py` only shows them because it configures logging at import time, which spawn re-imports.
- **Impact:** In local training the user never sees learner progress (`step=..., loss=...`) unless WandB is on. Worker failures show up only as bare tracebacks.
- **Fix:** Call one `setup_logging()` at the top of every process target, or use a `QueueHandler`/`QueueListener` back to the parent.

### R3-17: medium, bug. SIGTERM orphans child processes (`run-workers` and `train`)
- **Location:** `distributed.py:361-374` and `launcher.py:474-490`. Only `KeyboardInterrupt` is handled. The default SIGTERM action kills the parent without running `finally` or multiprocessing's atexit hooks.
- **Evidence:** VERIFIED-BY-RUNNING:
  - `kill -TERM` on the `run-workers` parent: "parent alive? no; former children still alive: 5749 ppid 295, 5750 ppid 295".
  - Same for `train`: the learner and worker survived with ppid 295 after `kill -TERM` on the main process. I killed these orphans by hand.
- **Impact:**
  - On bare-metal or systemd boxes, "removing a machine" leaves rollout processes running. They keep sending to learners until they hit their step budget.
  - In K8s the container teardown kills them, but with no graceful flush.
- **Fix:**
  - Install SIGTERM/SIGINT handlers in the parent that set `stop_event`.
  - Set `PR_SET_PDEATHSIG`, or have children poll `os.getppid()`.
  - Join, then terminate.

### R3-18: medium, design. Learner and worker lifecycles are not coordinated
- **Location:**
  - `distributed.py:192-193`: the learner stops at `total_train_steps`.
  - `distributed.py:342`: `per_worker_steps = total_timesteps // num_workers` on *each* host.
- **Evidence:** VERIFIED-BY-RUNNING. In `run_distributed_test`, the learner stopped at 15:19:56. The workers finished at 15:20:30, with every chunk after that dropped at debug level. In the 2-learner run, "workers alive after learners gone: yes".
- **Impact:**
  - Wasted CPU.
  - With K worker hosts, the run collects K× the budgeted data.
  - Under K8s, a finished learner is restarted by its Deployment, triggering R3-04, R3-05 and R3-06. Finished worker pods also restart.
- **Fix:** A single source of truth for run state (the hub). Workers run until told to stop, and learners report "done". Use K8s `Job` semantics, or have the learner idle-serve after it finishes instead of exiting.

### R3-19: medium, correctness. Checkpoint optimizer state is aliased to live tensors, producing torn checkpoints
- **Location:** `learner.py:125-135`. `state_dict_cpu` is cloned, but `algorithm.optimizer_state_dict` is `self._optimizer.state_dict()`, whose tensors are references to the live state.
- **Evidence:** VERIFIED-BY-RUNNING. After two more `opt.step()` calls, the saved snapshot's `exp_avg` had changed ("optimizer snapshot mutated by later steps: True").
- **Impact:**
  - In distributed mode the drainer thread writes `optimizer.pt` while training keeps mutating it, so the Adam moments do not match `model.pt`.
  - In local mode, sending through `mp.Queue` moves the live optimizer storages into shared memory (CPU), or shares them through CUDA IPC (GPU), which initializes CUDA in the launcher. That is SUSPECTED for the GPU case, not run here.
- **Fix:** `copy.deepcopy` the state to CPU, e.g. `{k: v.detach().cpu().clone()}` recursively, before enqueueing.

### R3-20: medium, fault-tolerance. Resume is partial even in local mode
- **Location:** `learner.py:64-77`, `appo.py:62-73`, `launcher.py:267-307`.
- **What is wrong:**
  - The LR scheduler is created fresh after resume (it restarts at step 0 over `total_train_steps`).
  - GradScaler state and the kickstart lambda are not restored.
  - Resuming from a `.pt` path sets `policy_version=0`.
  - Coordinator state (ELO, win-rate matrix, PFSP stats, agent pool), RNG states and worker match assignments are not persisted. The coordinator lives in the launcher's memory.
- **Evidence:** BY-READING.
- **Impact:** A resumed run is not equivalent to an uninterrupted one. In league mode, ratings and PFSP priorities restart from scratch.
- **Fix:**
  - Checkpoint a full learner state (model, optimizer, scheduler, scaler, kickstart, version, consumed env steps, RNG).
  - Have the coordinator snapshot its league state (json or sqlite) on every checkpoint and on shutdown.

### R3-21: medium, bug (packaging). The generated gRPC/protobuf code requires much newer versions than pyproject allows
- **Location:**
  - `transport/colosseum_pb2.py:5,12-18` validates protobuf runtime 6.31.1.
  - `colosseum_pb2_grpc.py:8-25` raises `RuntimeError` if grpcio < 1.78.0.
  - `pyproject.toml` allows `grpcio>=1.60`, `protobuf>=4.25`.
- **Evidence:** BY-READING (the explicit raise and validate calls). This environment had grpc 1.84, so it was not triggered here.
- **Impact:** Installs that satisfy the declared constraints fail at import of the distributed modules.
- **Fix:** Pin `grpcio>=1.78`, `protobuf>=6.31`, or regenerate the code with the minimum supported grpcio-tools in CI.

### R3-22: medium, security. Unauthenticated gRPC on all interfaces; weights can be poisoned; no decompression-size guard
- **Location:** `grpc_store.py:114` and `grpc_transport.py:74` (`add_insecure_port("[::]:port")`), `serialization.py:24-27, 63-66` (unbounded `lz4.frame.decompress`).
- **What is good:** All network deserialization uses `torch.load(weights_only=True)`. There is no `pickle` or `weights_only=False` on network data (VERIFIED-BY-READING; grep finds no `weights_only=False`).
- **What is wrong:**
  - Anyone who can reach the port can `PutWeights` for any agent (model poisoning, or rolling the version back to freeze workers, as in R3-04).
  - They can also flood the TrajectoryService with crafted chunks.
  - A 64 MiB lz4 frame can decompress to gigabytes, and nothing caps it.
  - `torch>=2.2` allows torch versions before 2.6, where `weights_only=True` has a known bypass (CVE-2025-32434).
- **Impact:** Moderate on a private competition cluster. Serious on cloud VMs with public IPs.
- **Fix:**
  - A `bind_address` option defaulting to the private interface.
  - A shared-secret token interceptor, or mTLS.
  - `torch>=2.6`.
  - Cap the decompressed size.
  - Use raw-buffer decoding (R3-14), which removes the `torch.load` parser from the network path.

### R3-23: medium, design. Workers and learners have no config or compatibility handshake
- **Location:** `distributed.py:142, 327`. Each role loads its own YAML independently.
- **Evidence:** BY-READING. `appo._prepare_batch` does `torch.stack` over chunks. A worker with a different `chunk_length` or obs shape crashes the learner. A different network definition silently produces log-probs from another architecture.
- **Impact:** Configs must be copied to every machine by hand and kept identical. One stale box can crash a learner.
- **Fix:**
  - Learners and workers fetch the run config from the hub, which stores a hash of it.
  - Workers send `(config_hash, network_hash)` on stream open, and the learner rejects mismatches with a clear error.

### R3-24: low, design/doc-mismatch. Dead abstractions and dead config
- **Location:** `transport/base.py`, `transport/local.py` (`LocalTransport`), `weight_store/shared_memory.py` (`SharedMemoryWeightStore`), `config.py` (`transport.mode`, `grpc_port`), `grpc_transport.py:33` (ignored `max_queue_size`).
- **What is wrong:**
  - None of the base classes or `LocalTransport` are used.
  - Local mode passes raw `mp.Queue`s and duck-typed adapters (`.put`/`.get_nowait`), and there is no weight store at all in local mode.
  - CLAUDE.md describes "shared memory for weights (zero-copy)". The `SharedMemoryWeightStore` that exists is `mp.Manager`-based (pickled bytes through a server process, not zero-copy) and unused.
- **Evidence:** BY-READING (grep).
- **Fix:** Delete them, or make the duck-typed "queue" protocol the official interface (`TrajectorySink`, `WeightSource`) with local and gRPC implementations.

### R3-25: low (suspected), bug. No deadlines or keepalive on any RPC
- **Location:** `grpc_transport.py:112`, `grpc_store.py:147-169`.
- **Evidence:** BY-READING.
- **Impact:** On a network partition without RST, or a frozen peer, `SendChunks` and `GetWeights` can block a worker or learner indefinitely, because no `timeout=` and no `grpc.keepalive_*` options are set.
- **Fix:** Per-call deadlines, keepalive pings, and `wait_for_ready` with a bounded timeout.

### R3-26: medium, deployment. Problems in the K8s and compose manifests
- **Location:** `deployment/k8s/*.yaml`.
- **What is wrong:**
  - Worker HPA on CPU utilization: rollout pods are always CPU-bound, so the HPA scales straight to `maxReplicas=16` regardless of learner capacity. The extra data is then dropped (R3-13).
  - Learner `Deployment` plus an RWO PVC with the default RollingUpdate strategy: the new pod cannot mount the PVC while the old one is alive (on another node).
  - A finished or crashed learner is restarted from scratch (R3-04, R3-05, R3-06).
  - The config is baked into the image (no ConfigMap).
  - One manifest is needed per agent.
  - The weight store has no PVC, so its state is lost on reschedule (R3-03).
  - `Dockerfile.worker` runs local `train`.
- **Evidence:** BY-READING.
- **Fix:**
  - A StatefulSet or `Recreate` strategy for learners.
  - Mount the config from a ConfigMap.
  - Scale the HPA on a custom metric (learner queue depth or `chunks_dropped`).
  - A Helm-style template with a list of agents.

### R3-27: low, bug. The global seed is applied before CLI overrides
- **Location:** `launcher.py:674-694`. Seeding uses `config.training.seed` from the YAML. `--set training.seed=X` is applied afterwards.
- **Evidence:** BY-READING.
- **Fix:** Apply the overrides first. Also seed `run-learner`, which never seeds at all.

### R3-28: medium, performance. The learner's input pipeline has no overlap
- **Location:** `learner.py:87-117`, `appo.py:87-118`.
- **What is wrong:**
  - Collecting chunks, stacking, H2D copy, training, weight push and checkpoint snapshot all run serially in one thread.
  - `pin_memory` calls `tensor.pin_memory()` on every new batch. That is a fresh `cudaHostAlloc` each time, which is often slower than a plain copy, and it gives no overlap because the result is consumed immediately.
  - In distributed mode, deserialization runs in gRPC threads that contend for the GIL with the loop.
- **Evidence:** BY-READING. There was no GPU available to measure.
- **Impact:** GPU idle gaps. These matter once R3-11 is fixed and batches get large.
- **Fix:**
  - A prefetch thread that assembles the next batch into reusable pinned buffers and issues a `non_blocking` copy on a side CUDA stream.
  - Push weights asynchronously.
  - Move deserialization into the prefetch thread, or into a process using raw buffers.

### R3-29: medium, test-gap. Tests cover only the happy path, and the scripts assert almost nothing
- **Location:** `tests/test_grpc.py`, `tests/test_distributed.py`, `tests/run_*_test.py`.
- **Evidence:** VERIFIED-BY-RUNNING:
  - `pytest test_grpc.py test_distributed.py`: 8 passed in 3.4 s.
  - `run_distributed_test.py`: "OK" in 2m38s.
  - `run_pipeline_test.py`: exit 0.
  - `run_scaled_test.py`: printed "Scaled test completed!" even though `Process-4` and `Process-5` crashed with `FileNotFoundError`.
  - No test covers store or learner restarts, worker death, multi-agent learner failure, batch size, weight staleness, message-size limits, SIGTERM, or final-checkpoint presence. R3-03 through R3-11 are all invisible to the suite.
  - `run_*_test.py` are not pytest tests, and they need cwd = repo root.
- **Fix:** Add pytest-marked integration tests for each of these failure modes, and assert final checkpoints, batch sizes and process exit codes.

### R3-30: low, performance. Local weight broadcast pickles N times; checkpoints go through the main process
- **Location:** `learner.py:209-215` (one `put_nowait` per worker queue, each pickled separately), `launcher.py:541-556`.
- **Impact:** O(num_workers) serialization per push. Fine for tiny networks, wasteful for large ones.
- **Fix:** A single shared-memory weight buffer per agent (seqlock/version) that all local workers read, which also fixes R3-10.

---

## Measured numbers

All on a loaded 8-core box (load average 20–35), `OMP_NUM_THREADS=1`.

### Chunk serialization (per chunk)

| Case | Wire bytes (current) | Serialize, current → raw numpy | Deserialize, current → raw numpy |
|---|---|---|---|
| T=32, 27 floats obs (tictactoe) | 2.5 KiB | 338 µs → 12 µs (28x) | 793 µs → 77 µs (10x) |
| T=256, 256 floats obs | 261 KiB (lz4 useless) | 592 µs → 27 µs (22x) | 930 µs → 85 µs (11x) |
| T=128, 8×32×32 sparse obs | 746 KiB (lz4 5.5x) | 27.3 ms (lz4-dominated) | 22.4 ms → 0.08 ms |

### Weights

| Params | Wire (lz4) | Learner put serialize | Client deserialize | Server cost per `GetWeights` |
|---|---|---|---|---|
| 0.2M | 0.75 MiB (ratio 1.002) | 1.1 ms | 1.2 ms | 2.2 ms |
| 4.2M | 16.0 MiB (ratio 1.000) | 83 ms | 47 ms | 125 ms |
| 33.6M | 128 MiB, **rejected with the default 64 MiB cap** | 612 ms | 390 ms | 1046 ms |

### gRPC send on localhost
- Stream per chunk (current): 18–28 ms per chunk, i.e. 36–56 chunks/s per worker process.
- One stream reused: 3.4–12 ms per chunk.
- With the learner queue full: **5.01 s per send**, then the chunk is dropped.

### End-to-end
- `run_distributed_test.py`: 2m38s total. The learner finished 75 steps in about 2 min under load. The workers then ran 34 s more, dropping every chunk.
- Effective learner batch with `batch_chunks=4`: 1.0–1.1 chunks (from `step`/`chunks` counters).
- Adding a second worker host: learner rate went from about 11 to about 30 steps/s (it was data-starved).

---

## Subsystem assessment

### What is good
- **The core idea of the adapters is sound.** `GRPCTrajectorySink`/`GRPCWeightSource`/`GRPCWeightSink` let the same `learner_process` and `rollout_worker_process` run over `mp.Queue` or gRPC. Because of that, rollouts on CPU boxes and per-agent learners on GPU boxes already function for plain self-play, and a worker host can join at runtime.
- **The weight path is pull-based by version** (`GetVersion` before `GetWeights`). In distributed mode this avoids R3-10's staleness bug.
- **Network deserialization uses `weights_only=True`.** No pickle crosses the network.
- **The learner reacts to SIGTERM cleanly.** It stops within about 1 s, stops its server and closes its channel.
- **The local launcher works for the happy path,** including runtime match refresh and checkpoint deltas to workers. That gives a good blueprint for a remote assignment protocol.

### What is structurally weak
1. **The coordinator is welded into the local launcher process.** Distributed mode therefore loses everything that makes Colosseum more than IMPALA: league, PFSP, history opponents, results and ratings.
2. **The failure model is "everything is always up".** Every RPC error, dead producer or exited learner crashes or hangs something. Exit codes lie. Nothing restarts, and resume is partial or absent.
3. **Identity and versioning are naive.** A bare int version, no incarnation, no config hash, checkpoint ids keyed only by version.
4. **The data and weight planes are written for tiny models.** Single-message weights with a 64 MiB cap, re-serialization on every pull, stream-per-chunk with synchronous sends, `torch.save` per chunk, a learner with no prefetch that trains on 1-chunk batches.
5. **Operations are hand-wired.** Ports, addresses and agent lists are set manually per host, the YAML is copied by hand, there are no metrics, child logs are lost, signals orphan processes, and the docs describe a different, non-working flow.

### Proposed target design (priority order, aimed at "CPU boxes for rollouts, GPU boxes for learners, dynamic machines and agents, simple setup")

1. **One "hub" service as the only address anyone needs** (`colosseum hub -c run.yaml --port 50051`). It hosts:
   - the run config (served to everyone, with a hash for checking)
   - the agent registry (add or remove agents at runtime with `colosseum agent add …` or a hub API)
   - membership with heartbeats and leases
   - the weight store: latest weights plus checkpoint blobs, stored once as bytes, streamed in pieces, versioned as `(incarnation, step)`
   - matchmaking (the current `Coordinator`)
   - result ingestion and ELO/PFSP
   - persisted league state (sqlite/json snapshots)

   This gives 3 commands for 3 machines and no per-agent ports in user config.
2. **Learner host** (`colosseum learner --hub A:50051 [--agents auto|x,y] [--devices cuda:0,cuda:1]`):
   - It registers its gRPC endpoint for each agent it owns. The hub assigns unowned agents, so learners for different agents can live on any number of GPU boxes and be added later.
   - It resumes automatically from the hub's latest checkpoint (model, optimizer, scheduler, scaler, version, consumed steps).
   - It runs a receive/prefetch thread with fixed-size batches (fixes R3-11) and pushes weights asynchronously.
   - It takes snapshot-safe checkpoints (fixes R3-19) and uploads them to the hub, so all workers can use historical opponents.
   - It emits metrics (fixes R3-15).
3. **Worker host** (`colosseum workers --hub A:50051 [--procs auto]`):
   - A per-machine supervisor that restarts crashed rollout processes and handles SIGTERM properly.
   - One machine-local weight and checkpoint cache in shared memory, pulled once per machine per version (fixes R3-12's N×).
   - Per process: assignments leased from the hub (the existing `WorkerCommand` format), results reported back (fixes R3-02), and chunks sent over a **persistent** stream per learner from a background sender with a bounded drop-oldest outbox and counters (fixes R3-13).
   - Learner endpoints are discovered from the hub, so agent moves and restarts are transparent.
4. **Wire format:** a header plus raw contiguous buffers, lz4 only for compressible observations (R3-14). Weights streamed in pieces, optionally in bf16. A config and network hash on stream open (R3-23).
5. **Reliability defaults:** `wait_for_ready`, deadlines, keepalive, retry with backoff on every client. Learners never crash on store or hub outages; they keep the last payload and re-push it. Nonzero exit codes on real failures. Integration tests for each failure mode in R3-03 through R3-11.
6. **Local mode = the same components with the hub in-process** and shared-memory transports. One code path, so single-machine runs exercise the distributed logic. Remove the mp.Queue tensor fd-sharing that causes R3-07 and R3-08.
7. **Security and deployment:**
   - Bind-address and token/mTLS options; `torch>=2.6`.
   - K8s: hub StatefulSet with a PVC, learner StatefulSet (GPU) with one replica per learner host, worker Deployment.
   - HPA on hub-reported learner-starvation metrics.
   - Config from a ConfigMap.
