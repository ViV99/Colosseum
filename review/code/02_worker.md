# R2 — Rollout worker, environments, trajectory data model

Scope: `worker/rollout_worker.py`, `envs/{base_env,vec_env,subproc_vec_env}.py`, `core/{types,outcomes,action_spec}.py`, plus the parts of `launcher.py` / `distributed.py` / `learner.py` that feed workers.
Repro scripts are in `scratchpad/r2_worker/` (`common.py` has a deterministic `StepEnv`: obs = `[episode, t, player]`, reward = `100*episode + t + 0.1*player`).
The existing relevant tests pass (78 passed: vec_env, subproc, multi_agent, recurrent, performance, integration, composite). None of the bugs below is covered by a test.

Throughput numbers were measured while other agents were running: load average was 20–30 on 8 cores. Absolute values are noisy. The ratios are large enough to rely on.

---

## Findings

### R2-01 — CRITICAL — bug — recurrent (LSTM/GRU) training crashes when it uses chunks from the real worker
- **Location:** `worker/rollout_worker.py:106-109, 155-158, 219-220`; `algorithms/appo.py:157-164`; `core/types.py:39-42`
- **What:** The worker stores the hidden state per slot as `[num_layers, 1, H]`. That comes from `initial_hidden(1)` and the slice `new_hidden[0][:, j:j+1, :]`. It is put into `chunk.lstm_hidden` unchanged. `TrajectoryChunk` documents `[num_layers, hidden]`, and APPO does `torch.stack([c.lstm_hidden[0] ...], dim=1)`, which produces `[L, B, 1, H]` (4-D), and the LSTM rejects it. The APPO recurrent tests build chunks by hand with `h.squeeze(1)` (`tests/test_recurrent.py:231`), so the worker→learner path is never tested.
- **Evidence (VERIFIED-BY-RUNNING, `repro_lstm.py`):**
  ```
  worker lstm_hidden h shape: (1, 1, 8)
  APPO train_step FAILED: RuntimeError For batched 3-D input, hx and cx should also be 3-D but got (4-D, 4-D) tensors
  ```
- **Impact:** The "LSTM/GRU support" feature listed as DONE does not work end-to-end. The learner process dies on its first batch.
- **Fix:** Store `h[:, 0, :]` in `set_lstm_init`, or `squeeze(1)` in `_build_chunk`. Validate the shape in the `TrajectoryChunk.__post_init__` dataclass. Add an integration test: worker produces chunks → `APPO.train_step`.

### R2-02 — HIGH — correctness — turn-based `info["active"]` path drops the inactive player's terminal reward and `done`, so episodes merge in that player's data
- **Location:** `rollout_worker.py:496-497` (a transition is appended only if `slot_active`)
- **What:** A slot's transition is recorded only on steps where it is active. Rewards given on inactive steps are discarded, and so is `done` when the episode ends on an opponent's move. The usual case is "opponent makes the winning move → I get −1 and the game ends". The loser never sees −1 and never gets `done=1`. Its next recorded step is from the next episode, with no boundary in between.
- **Evidence (VERIFIED-BY-RUNNING, `repro_alignment.py`, case C):** Player 1 loses every game on player 0's move:
  ```
  p1 obs(ep,t)=[(0,1),(0,3),(1,1),(1,3)] r=[1.1, 3.1, 101.1, 103.1] d=[0,0,0,0]
  p1 obs(ep,t)=[(2,1),(2,3),(3,1),(3,3)] r=[201.1, 203.1, 301.1, 303.1] d=[0,0,0,0]
  ```
  The −1 is never recorded, and episodes 0→1→2→3 are chained without any `done`.
- **Impact:** For any turn-based game that uses the documented `active` mechanism, the policy only learns from winning, and V-trace/GAE bootstrap across episode boundaries. The feature is undocumented in README and has no tests.
- **Fix:** Keep a per-slot "pending" reward accumulator. Add rewards from inactive steps to the slot's last recorded transition (`buf._rewards[cursor-1] += r`). On episode end, set `buf._dones[cursor-1] = 1` for every collecting slot, whether active or not. If the last transition was already sealed into a chunk, defer sealing by one step (see R2-11) so this can still be patched. Add tests.

### R2-03 — HIGH — bug — `ActionSpec` orders Tuple/MultiDiscrete components lexicographically ("0","1","10","11","2",…), so actions and masks reach the wrong dimensions when there are more than 10 components
- **Location:** `core/action_spec.py:107, 111-113, 130` (`for name in sorted(named.keys())`), `decode` 194-208, `flatten_mask` 258-259. `CompositeDist` sorts the same way (`networks/distributions.py:133`).
- **What:** Tuple and MultiDiscrete components are named `str(i)` and then sorted as strings. `decode` returns values in component order, so env dimension 2 receives the sample from head "10". A flat ndarray mask in natural env order is passed through unchanged, but it is sliced using the lexicographic offsets.
- **Evidence (VERIFIED-BY-RUNNING, `repro_actionspec.py`, `MultiDiscrete([2..13])`):**
  ```
  component order: ['0','1','10','11','2',...]
  num_categories per flat slot: [2, 3, 12, 13, 4, ...]
  decode(flat=[0..11]) -> [0,1,2,...]   # env dim 2 (n=4) gets head '10' (n=12) -> values up to 11
  flatten_mask(ndarray) segment for comp '2': [T,T,T,T] (should be [T,F,F,F])
  ```
- **Impact:** This breaks the typical competition action space, e.g. MultiDiscrete per unit or per cell in Lux AI. Envs receive out-of-range or wrong actions and masks are applied to the wrong heads, with no error raised. Tests only cover two or three dimensions.
- **Fix:** For tuple/multi_discrete, keep the natural order (no `sorted`), or use zero-padded names (`f"{i:04d}"`). Make `CompositeDist` follow the same rule; ideally it should take its order from the `ActionSpec`. Add a test with 12 or more dimensions.

### R2-04 — HIGH — performance — worker processes never call `torch.set_num_threads`, so intra-op threads oversubscribe the CPU
- **Location:** `rollout_worker.py` (no thread configuration anywhere), `launcher._worker_target`, `distributed._dist_worker_target`. There is no `set_num_threads` anywhere in `src/`.
- **What:** Each worker process uses PyTorch's default intra-op pool of one thread per core. N workers on one machine means N×cores spinning threads, all running tiny batches.
- **Evidence (VERIFIED-BY-RUNNING, `bench.py tic_tac_toe 8 envs`, same machine, back-to-back):**
  ```
  threads=8 (default): 44 env-steps/s
  threads=2:         1158 env-steps/s
  threads=1:         4074 env-steps/s
  ```
  In `repro_reassign_waste.py` the same workload took more than 5 minutes at 214% CPU with default threads, and 15 s with 1 thread.
- **Impact:** Throughput drops by one to two orders of magnitude on any machine that is shared, or that runs more than one worker, which is the normal deployment.
- **Fix:** In the worker entry point, call `torch.set_num_threads(cfg.rollout.worker_torch_threads or 1)` and `torch.set_num_interop_threads(1)`, and set `OMP_NUM_THREADS=1` in the environment of spawned children. Do the same in SubprocVecEnv children.

### R2-05 — HIGH — bug — the weight queue is FIFO with maxsize 2, and the learner drops pushes when it is full, so workers load the oldest weights rather than the newest
- **Location:** `launcher.py:34` (`_WEIGHT_QUEUE_SIZE = 2`), `learner/learner.py:196-215` (`put_nowait`, skip on `Full`), `rollout_worker.py:726-739` (drain, keep the last item)
- **What:** Between two worker syncs (default every 5 s), the learner pushes every `weight_push_interval` train steps. The first two pushes fill the queue and every later push is dropped. When the worker drains the queue it gets push #2, which is about one sync interval old, not the current weights.
- **Evidence (VERIFIED-BY-RUNNING, `repro_weightlag.py`):** `learner latest version = 50; worker loaded version = 2`.
- **Impact:** Policy lag is systematically one extra sync interval on top of what the design intends, and it gets worse as the learner gets faster. This directly hurts APPO (larger importance ratios, more clipping) and wastes the configured sync frequency.
- **Fix:** Use "latest wins" semantics. Options: a size-1 slot where the learner does a non-blocking get to evict the old item before putting the new one; a shared-memory weight store plus a version counter (that store already exists as `SharedMemoryWeightStore`); or a single `mp.Value` version plus a `Manager` dict. Push a version number and fetch the payload only if it is newer.

### R2-06 — HIGH — bug — `WorkerCommand`s are coalesced, which loses their checkpoint deltas; the worker then silently plays "latest" in a slot labelled as a checkpoint
- **Location:** `rollout_worker.py:166-176` (`_drain_commands` keeps only the newest command), `179-197`, `475` (`agent_nets.get(net_id, agent_nets[LATEST_NETWORK_ID])`); `launcher.py:619-636` (marks a checkpoint as sent *before* `put_nowait`, and drops the command on `Full`)
- **What:** Checkpoint state_dicts are sent only once, as a delta. If two refreshes are queued (the worker is blocked in `tq.put` because the learner is slow, which is a normal backpressure state), the earlier command's `new_checkpoints` are thrown away. If the queue is full, the command is dropped, but the launcher has already recorded the checkpoint as sent. Either way the worker never receives that checkpoint. Later assignments refer to it, the lookup silently falls back to the latest network, and the result is reported as `agent:ckpt_vX`.
- **Evidence (VERIFIED-BY-RUNNING, `repro_commands.py`):**
  ```
  worker pool after draining 2 commands: ['latest'] ; staged slot nets: [['latest', 'ckpt_v100']]
  no crash, result keys reported: ['a:latest', 'a:ckpt_v100'] <- slot 1 actually ran 'latest' weights
  ```
- **Impact:** The distribution of historical opponents and the self-play statistics are quietly corrupted, and nothing logs it.
- **Fix:** In `_drain_commands`, merge `new_checkpoints` across all drained commands and use the latest maps. In the launcher, mark checkpoints as sent only after a successful put, or make the worker request missing checkpoints by id. Raise an error or log loudly on an unknown `net_id`; never fall back silently.

### R2-07 — HIGH — design gap vs owner priorities — scripted bots, external frozen agents, and checkpoints with a different architecture cannot be placed in a worker slot
- **Location:** `rollout_worker.py:290-315, 474` (`networks_by_agent` is keyed only by *trainable* agent ids; checkpoint networks are built with that agent's current factory)
- **What:** `AgentPool` defines scripted and frozen agents, but the worker has no code path for them. An unknown `agent_id` in `slot_agent_map` raises `KeyError`. A frozen checkpoint has to load into the current architecture of a trainable agent (`load_state_dict`, strict), so you cannot fight an older-architecture version or a separately trained model. Rule-based bots are not possible at all. No matchmaker generates scripted or frozen slots.
- **Evidence (VERIFIED-BY-RUNNING, `repro_commands.py`):** `scripted/frozen-external agent in slot -> KeyError 'scripted_bot'`.
- **Impact:** The "flexible final arena (different weights/architectures, scripted bots mixed in)" requirement is not met in training. Only `eval.py` might cover parts of it.
- **Fix:** Introduce a `Policy` abstraction on the worker side: `act(obs_batch, masks, hidden) -> actions, logp, values`. Implementations: `TorchPolicy(net)`, `ScriptedPolicy(callable(raw_obs, info))`, and `FrozenPolicy` built from a per-checkpoint config stored with the checkpoint (an architecture spec plus weights). Key the pool by an opaque `policy_id`, not `(trainable_agent, ckpt)`.

### R2-08 — MEDIUM — bug — `MatchResult` is keyed by `"agent:network"`, so slots sharing a key overwrite each other
- **Location:** `rollout_worker.py:648-653`; `coordinator.py:149-154`
- **What:** If two slots share an agent and a network (self-play latest vs latest, or a PFSP arena with `num_players>2`, which duplicates agents as `[A,B,A,B]`), the dict keeps only the last slot's outcome and reward. The coordinator's "average across slots of the same agent" logic is effectively dead code.
- **Evidence (VERIFIED-BY-RUNNING, `repro_alignment.py` C):** In the turn-based self-play game, p0 has rank 1 and p1 has rank 2, but the result is `{'a:latest': 0.0}`. That is a single entry, and it holds the loser's value.
- **Impact:** In 3+-player arenas, ELO and win rates come from an arbitrary slot (in FFA the winning A slot can be recorded as a loss). Seat/position information is lost, so first-mover bias cannot be analysed.
- **Fix:** Report a list of per-slot records: `[{slot, agent_id, network_id, policy_version, outcome, reward, rank}]`. Let the coordinator aggregate.

### R2-09 — MEDIUM — correctness — truncation is treated as termination; the terminal observation is collected but never used
- **Location:** `rollout_worker.py:490, 515, 531` (`done = terminated or truncated`; `bootstrap_val = 0.0 if done`); `vec_env.py:139` (`terminal_observation` is stored but has no consumer); `TrajectoryChunk` has a single `dones` field
- **Evidence (VERIFIED-BY-RUNNING, case B):** For a time-limit episode, `dones=[0,0,0,1] boot=0.000`. The mid-chunk V-trace target also uses `not_done=0` there.
- **Impact:** Biased value targets for envs with step limits, or when a match is cut short (e.g. a `max_ticks` cap reported as truncated).
- **Fix:** Add `terminated` and `truncated` to the chunk. For truncated steps, bootstrap `V(terminal_observation)`: either the worker computes it, or a `truncation_values` tensor is stored. In V-trace, use `discount = gamma * (1 - terminated)` while still cutting the trace at both.

### R2-10 — MEDIUM — correctness — recurrent policy combined with `active` flags: the worker's hidden state advances on inactive steps that the learner never replays
- **Location:** `rollout_worker.py:473-479` (all slots run inference and update `hidden_states`) vs `496-497` (only active steps are recorded)
- **Evidence (VERIFIED-BY-RUNNING, `repro_lstm.py`, after reshaping around R2-01), max |learner logp − behaviour logp| with identical weights:** simultaneous env `2.4e-07`, turn-based env `9.6e-02`.
- **Impact:** Importance ratios differ from 1 even fully on-policy, which biases the clipped objective.
- **Fix:** Either update the hidden state only on active steps (mask the hidden update per slot), or record every step and pass a `valid` mask to the learner so the loss is masked but the RNN is unrolled over all steps.

### R2-11 — MEDIUM — performance + correctness — the bootstrap value costs an extra unbatched forward pass per chunk and is computed with the worker's stale network
- **Location:** `rollout_worker.py:520-531`
- **What:** Each sealed chunk calls `bootstrap_net.act(next_obs)` on a batch of one. That samples an action and computes log_prob, which are thrown away, and it passes no mask. The next loop iteration computes `V(next_obs)` for that slot anyway, in a batch. The value also comes from the behaviour network, while V-trace uses learner-side values for every other step (`vtrace.py:65`).
- **Evidence (VERIFIED-BY-RUNNING, cProfile):** tic_tac_toe (T=32, 16 slots): 1496 `multinomial` calls for 1000 loop iterations, so 33% of forward passes are bootstrap calls. Chase (T=20, 16 slots): 1800 vs 1000, so 44%.
- **Fix:** Defer sealing a full buffer by one step and fill the bootstrap from the batched values of the next iteration. Better still, store `obs[T]` in the chunk (T+1 observations) and let the learner compute the bootstrap with current parameters, as Sample Factory and IMPALA do. That also makes recurrent bootstrapping consistent.

### R2-12 — MEDIUM/LOW — data waste — partial buffers are discarded when a slot's agent or collect flag changes at re-assignment
- **Location:** `rollout_worker.py:573-577`
- **Evidence (VERIFIED-BY-RUNNING, `repro_reassign_waste.py`, 8 envs, episodes of 9 steps, T=128, random latest/ckpt opponent per refresh):** refresh every 200 iterations discards 5.7% of collected transitions; every 20 iterations discards 33.4%. With the default 30 s refresh the loss is smaller, but up to T−1 steps per flipped slot per refresh. With long episodes and T=512 it is up to 511 steps per slot.
- **Fix:** Seal and send the partial chunk instead of discarding it. The learner already handles `dones` inside chunks; it would need padding plus a `valid` mask, or variable-length chunks. Alternatively, only re-assign slots whose buffers are empty.

### R2-13 — MEDIUM — memory — the worker's checkpoint network pool is never evicted; checkpoints are re-read from disk on every refresh
- **Location:** `rollout_worker.py:185-193` (add only); `launcher.py:222-250` (`_derive_worker_configs` builds a fresh dict and calls `checkpoint_manager.load` for every referenced checkpoint, for every worker, on every refresh)
- **What (BY-READING):** Each worker keeps every checkpoint network it has ever been sent, even after FIFO pruning removes them from disk. The monitor loop does `num_workers × distinct_ckpts` disk loads every 30 s, even though only deltas are sent.
- **Impact:** Worker RSS grows without bound over long league runs (number of checkpoints × model size, per worker). Monitor-loop latency also grows.
- **Fix:** Use an LRU or reference-counted pool limited to the networks referenced by the current and pending maps. Cache loaded state_dicts in the coordinator. Better: let workers fetch checkpoints by id from the weight store or a shared filesystem.

### R2-14 — MEDIUM — bug — `VectorEnv` auto-reset merges reset info on top of the terminal-step info, so stale keys (`action_mask`, `active`, `rank`, `outcome`, …) leak into the next episode's first decision
- **Location:** `vec_env.py:137-147` (`info_k[p].update(new_info[p])`)
- **Evidence (VERIFIED-BY-RUNNING, `repro_vecenv_infoleak.py`):** After the reset, the pre-step info used for episode 1, t=0 still contains `'rank': 1`, `'outcome': 0.123`, and the terminal-state `action_mask` `[False, True, True, True]`.
- **Impact:** If the env omits `action_mask` from the reset info, the agent acts in the new episode under the previous episode's terminal mask. An all-False terminal mask crashes the worker (R2-15).
- **Fix:** For done envs, return `new_info[p]` as the info and keep the old one only under `terminal_info` / `terminal_observation`, i.e. Gymnasium's `final_info` convention.

### R2-15 — MEDIUM — robustness — a NaN observation or an all-False action mask kills the worker process, and nothing restarts it
- **Location:** `rollout_worker.py:148-151` (no validation); torch `Categorical` `validate_args`; `launcher.py:534-539` only stops when *all* workers are dead
- **Evidence (VERIFIED-BY-RUNNING, `repro_nan.py`):** Both cases give `ValueError Expected parameter logits ... to satisfy the constraint`. An all-False mask is a natural convention for inactive players in turn-based games, and inference runs for inactive slots too.
- **Impact:** A single bad env state permanently removes one worker. Throughput degrades without any notice.
- **Fix:** Before inference, replace masks that are all-False with all-True, or skip inactive slots entirely. Check `np.isfinite(obs)` and log or quarantine the env. Wrap the env step in try/except and reset that env. The launcher should restart dead workers.

### R2-16 — MEDIUM/LOW — performance/robustness — `SubprocessVectorEnv` design
- **Location:** `envs/subproc_vec_env.py`
- **(a) Exceptions are not forwarded (VERIFIED):** `_worker_loop` only catches `EOFError` and `KeyboardInterrupt`. An env exception kills the child, and the parent sees `EOFError()` with no context. The `_recv` code path that re-raises `BaseException` (`:353-359`) is dead code.
- **(b) Strict lockstep with pickled pipes:** The parent sits idle while the children step, and the children sit idle while the parent runs inference. There is no double-buffering or shared memory. Measured, 2 children: tic_tac_toe 591 env-steps/s vs 1802 for sync; chase 630 vs 691 for sync.
- **(c) Heavy children (VERIFIED):** Each child unpickles `partial(colosseum.launcher._create_env)`, which imports `launcher` and therefore torch and the coordinator. That is about 240 MB RSS per child, even for tic-tac-toe.
- **(d)** Default `num_workers = min(num_envs, cpu_count)`: every rollout worker spawns up to cpu_count children, which oversubscribes the machine (combine with R2-04).
- **Fix:** Send `(type(e), traceback string)` back to the parent. Use shared-memory obs, reward and done buffers, with only actions and small info dicts going over the pipe. Split the envs into two halves and alternate stepping and inference between them, as Sample Factory does. Move `_create_env` into a torch-free module. Default `subproc_workers` to a small number.

### R2-17 — MEDIUM — correctness/warm start — workers start acting before they have the learner's weights, and their init is not seeded
- **Location:** `rollout_worker.py:317-319` (one non-blocking `_sync_weights` call), `371-376` (seeding happens after the networks are built)
- **What (BY-READING):** If the learner has not pushed its initial weights yet (it is still importing, initialising CUDA, or loading `resume_from` or a kickstart teacher), the worker collects with a randomly initialised network for up to `weight_sync_interval` seconds. Those chunks are labelled version 0, the same as the learner's real version 0. With a BC warm start, the early batches are random-policy data that look on-policy.
- **Fix:** Block on startup until each agent's weights arrive, with a timeout and a clear log message. Seed before building the networks.

### R2-18 — LOW/MEDIUM — data model — `behavior_policy_version` is never consumed, and it is stamped at seal time
- **Location:** `rollout_worker.py:533-536`; no reader in `learner/` or `algorithms/` (grep)
- **What (BY-READING):** A chunk that spans a weight sync is labelled with the newer version. Nothing computes policy lag, discards over-stale chunks, or logs it.
- **Fix:** Record the version per step (an int32 `[T]`, cheap) or at least the min and max. Have the learner log `policy_lag = learner_version − behavior_version` and optionally drop chunks above a threshold.

### R2-19 — MEDIUM — design — no per-player termination: "any player terminated" ends the whole match
- **Location:** `vec_env.py:126-129`; `base_env.py:74-75`
- **What (BY-READING):** In an N-player FFA, eliminated players must stay in the match with dummy observations until the game ends. The worker keeps recording their transitions, or, if `active=False`, drops their later rewards such as a final ranking bonus (see R2-02). There is no per-player `done`, so an eliminated player's trajectory cannot end at elimination with its final reward.
- **Fix:** Per slot: done, an `alive` mask, and reward accumulation (as in R2-02). Let the env emit `terminated[p]` for a single player without resetting the match.

### R2-20 — MEDIUM — design/performance — observations must be one fixed-shape array and are forced to float32
- **Location:** `rollout_worker.py:72` (buffer is always `float32`), `134` (`.float()`); `vec_env.py:42-48` (`np.array` stacking)
- **What (BY-READING):** Dict or Tuple observation spaces are not supported. Lux-style "image + global vector + per-unit list" observations have to be flattened by the user. Agents with different encoders still receive the same raw tensor; there is no per-agent obs preprocessing hook. uint8 observations are inflated 4× in buffers, chunks and transport.
- **Fix:** Use an ObsSpec codec mirroring `ActionSpec`: a dict of named arrays with their native dtypes, in TensorDict-like chunks. Add an optional per-agent `obs_transform(raw_obs, info)` that runs in the worker.

### R2-21 — LOW — design — worker inference is CPU-only, with one tiny forward pass per (agent, network) group
- **Location:** `rollout_worker.py:453-479`
- **What (BY-READING):** Networks are never moved to a device, so a worker cannot use a local GPU or accelerator. With PFSP and a pool of checkpoints, every distinct checkpoint gets its own forward pass with batch size 1–2 on every step.
- **Fix:** Add a `rollout.inference_device` option. Batch across workers through a per-machine inference server (SEED-style) as an option. Prefer fewer, larger groups (e.g. assign one checkpoint per env across all its slots).

### R2-22 — LOW — ux — `total_timesteps` means different things for workers and learners
- **Location:** `launcher.py:356-361, 454`
- **What (BY-READING):** Workers stop after `total_timesteps / num_workers` **env** steps. Learners stop after `total_timesteps / (T·batch_chunks)` **chunks**, and a 2-player self-play env step yields two agent transitions. With frozen opponents (non-collecting slots) or discarded partial buffers, learners finish early or never reach their step count. The run then ends via the "all workers dead" fallback.
- **Fix:** Define the budget in agent-steps consumed by the learner, and let the learners drive termination.

### R2-23 — MEDIUM — performance/robustness (cross-reference R3) — the distributed worker sends each chunk as a blocking unary gRPC call in the env loop, with no deadline
- **Location:** `distributed.py:58-73`; `transport/grpc_transport.py:103-111`; weight polling at `distributed.py:92-100`
- **What (BY-READING):** Each sealed chunk is serialised, compressed with lz4, and sent via a synchronous `SendChunks(iter([proto]))` on the main loop. A hung learner blocks the worker forever. Failed sends are dropped but still counted in `chunks_sent`. Distributed workers get no `results_queue` and no `command_queue`, so there is no ELO and no historical or PFSP opponents across machines.
- **Fix:** Use a background sender thread with a bounded queue, a persistent stream, and deadlines. Run the coordinator as a service for results and commands.

### R2-24 — LOW — correctness/test-gap — the TicTacToe example exercises neither `active` nor `action_mask`; its `reset` reseeds numpy's global RNG
- **Location:** `examples/tic_tac_toe/env.py:51-53, 63-108`
- **What (BY-READING):** The inactive player's ignored moves are recorded as training data, which adds noise. Illegal moves are allowed and end the game. `np.random.seed(seed)` inside `reset` resets the RNG for the whole worker. The flagship example therefore does not demonstrate the turn-based handling that R2-02 shows is broken.
- **Fix:** Emit `active` and `action_mask` in the example and use a local `np.random.Generator`.

### R2-25 — LOW — robustness — the worker has no try/finally; shutdown drops partial results silently
- **Location:** `rollout_worker.py:430-617`, `661-664`
- **What (BY-READING):** On an exception, `vec_env.close()` and `cancel_join_thread()` are skipped, so the process can hang on exit while flushing a queue. The launcher then `terminate()`s it after 3 s. `results_queue.put_nowait` drops results silently when full, and episodes cut by shutdown are never reported. Acceptable, but drops should be counted and logged.
- **Fix:** Wrap the loop in try/finally. Count and log dropped results.

### R2-26 — test-gap
Missing tests:
- Chunk content alignment (obs[t] with action[t], reward[t], done[t]; auto-reset; bootstrap).
- Worker → APPO recurrent end-to-end (R2-01).
- `active` flags (R2-02, R2-10).
- Composite spaces with 11+ components (R2-03).
- Command coalescing and unknown-network handling (R2-06).
- Duplicate-key results (R2-08).
- Truncation (R2-09).
- Info leak (R2-14).
- All-False masks and NaN observations (R2-15).
- SubprocVecEnv exception propagation (R2-16a).

### R2-27 — doc-mismatch
- `CLAUDE.md:396-408` lists as "Remaining Open Items" a distributed launcher (#3), SubprocessVectorEnv (#6) and observation normalization (#11). All three exist: `distributed.py`, `envs/subproc_vec_env.py`, `networks/normalization.py`.
- `core/types.py:39-42` documents `lstm_hidden` as `[num_layers, hidden_size]`; the worker emits `[L, 1, H]` (R2-01).
- README documents action masks, but not the `info["active"]` turn-based convention or the `rank`/`outcome` terminal-info convention in `core/outcomes.py`.
- README:375 says composite masks may be "a flat mask" without saying the flat layout is lexicographic (R2-03).
- `MatchResult` docstring (`types.py:148-151`) says outcomes are keyed by `agent_id`; they are actually keyed by `agent_id:network_id` (R2-08).

### R2-28 — LOW (cross-reference R4) — seat bias from the matchmaker, as seen by the worker
- `SelfPlayMatchmaker`, `PFSPMatchmaker._solo_match` and `_arena_match` always put the primary agent in slot 0 (`matchmaker.py:106, 205, 222-225`). In turn-based games the primary agent always moves first.
- Only `agent_ids[0]` (`launcher.py:421, 607`) gets historical checkpoints. Other trainable agents never face their own history.

---

## Checked and correct (no finding)
- **Alignment within an episode** (VERIFIED, case A): `obs[t]`, `action[t]`, `reward[t]` (the reward produced by that action), `done[t]` and `value[t]` line up. After auto-reset the reset observation correctly becomes the next row; it does not overwrite the terminal row. Rewards go to the right player. The bootstrap is `V(next_obs)` for a chunk that ends mid-episode and 0 for one that ends at a terminal step.
- **Aliasing of pre-allocated buffers:** `_build_chunk` copies every array (`.copy()`) and the LSTM init is `.clone()`d, so reusing the buffer after `tq.put` is safe. With `queue.Queue` in the distributed path the chunk object is shared but immutable after sealing.
- **Hidden-state reset at done** and capture at chunk start match `evaluate_actions_recurrent(dones_seq=...)` (error 2.4e-7 for simultaneous envs once R2-01 is worked around).
- **Episode results** are reported with the slot maps that were in effect *before* the re-assignment for that env is applied.

## Measured throughput (1 worker in-process, chunks discarded; machine load avg 20–30 on 8 cores)

| Env | envs | torch threads | vec_env | env-steps/s | agent-steps/s |
|---|---|---|---|---|---|
| tic_tac_toe | 8 | 8 (default) | sync | **44** | 89 |
| tic_tac_toe | 8 | 2 | sync | 1158 | 2316 |
| tic_tac_toe | 8 | 1 | sync | 1500–4074 (noisy) | 3000–8150 |
| tic_tac_toe | 32 | 1 | sync | 3945 | 7890 |
| tic_tac_toe | 8 | 1 | subprocess(2) | 591 | 1183 |
| chase (Dict action) | 8 | 1 | sync | 691 | 1382 |
| chase | 8 | 1 | subprocess(2) | 630 | 1260 |
| space_miners | — | — | — | not measured: `Box2D` is not installed and the scope forbids installing packages | |

Where the time goes (cProfile, tic_tac_toe, 8 envs, 1 thread, 5.3 s total):
- Inference is about 43% (2.3 s), and a third of the forward passes are per-chunk bootstrap calls (R2-11).
- Inside inference, `multinomial`, `logsumexp`, `linear` and torch `Distribution` argument validation (`constraints.check`) dominate. Validation could be disabled with `validate_args=False`.
- The env is about 26%, half of it the example's `_get_obs`.
- The worker's own Python loop is about 0.4 s tottime, plus `append` 0.1 s.
- Episode-result reporting is about 4%: `outcomes_from_rewards` runs on every episode end.

For chase, the env's own step is about 20%, and composite distribution construction and validation is a large share of inference. Per-step Python overhead in the worker (rebuilding the group dict, per-slot loops, mask list → array) is roughly 50–100 µs per slot. That only matters once threads are fixed and envs are cheap.

---

## Subsystem assessment

**What is good**
- The single-process data path is simple and readable. Pre-allocated buffers are copied correctly at seal time, so there are no aliasing bugs.
- Within-episode alignment and auto-reset handling are correct, and hidden-state resets are consistent with the learner's reset-aware unroll.
- Inference is batched per (agent, network). Multi-agent routing by `slot_agent_map` works, and chunks never mix agents: a buffer is reset when its slot's agent or collect flag changes.
- Weight transport and queues are injected as duck-typed objects, which is how the gRPC adapters drop in without changing the worker.
- `ActionSpec`/`CompositeDist` is a good abstraction (after the ordering fix), and `outcomes.py` correctly prefers authoritative env signals over shaped reward.

**What is structurally weak**
1. **The data model is too thin for the games this framework targets.** There is a single `done` (no terminated/truncated split), no `obs[T]`, no per-step validity or `active` mask, no per-player alive state, a single policy version per chunk, and array-only float32 observations. Turn-based, elimination and time-limited games all lose information (R2-02, R2-09, R2-10, R2-19, R2-20).
2. **The worker's "policy pool" is hard-wired to trainable agents' torch networks.** There are no scripted bots, no externally trained or other-architecture frozen agents, no eviction, and missing networks silently fall back to latest (R2-06, R2-07, R2-13). This directly blocks the owner's "flexible final arena" priority.
3. **Control-plane messages assume lossless delivery but use lossy queues.** Weights are FIFO with drop-newest (R2-05), commands are coalesced and checkpoints are delta-only (R2-06), and results are dropped when full (R2-25). Messages carrying state should be idempotent snapshots, e.g. "here is the full set of policy ids you should have, fetch what you are missing".
4. **The performance basics are missing:** thread pinning (R2-04), deferred bootstrap (R2-11), and a shared-memory or double-buffered SubprocVecEnv (R2-16).
5. **Fault handling:** one bad observation or mask kills a worker permanently (R2-15). Subprocess env errors lose their tracebacks (R2-16a).

**Recommended changes, in priority order**
1. Fix R2-01: squeeze the hidden state and add a worker→APPO recurrent test. This is a one-line fix for a feature that is currently dead.
2. Fix R2-04: set torch to 1 thread per worker and per subprocess child. This gives a 10–90× throughput gain on shared machines.
3. Fix R2-03: natural ordering of Tuple/MultiDiscrete components in both `ActionSpec` and `CompositeDist`.
4. Fix R2-05: "latest-wins" weight delivery with a version check.
5. Fix R2-06: merge drained commands and fail loudly on unknown `net_id`; then make checkpoint distribution pull-based by id (also fixes R2-13).
6. Redesign the per-slot transition model:
   - pending-reward accumulation;
   - `done` written on episode end for every collecting slot;
   - a terminated/truncated split with `obs[T]` stored;
   - an optional `valid` mask instead of skipping steps;
   - seal partial chunks instead of discarding them (R2-12).
   This fixes R2-02, R2-09, R2-10 and R2-12 and enables R2-19.
7. Introduce a worker-side `Policy` interface (torch, scripted, frozen-with-own-architecture) keyed by `policy_id`, and per-slot `MatchResult` records (R2-07, R2-08).
8. Fix the `VectorEnv` info contract (R2-14); sanitize masks and NaN observations, with env-level try/except and worker restart in the launcher (R2-15).
9. Defer bootstrap to the next batched inference, or move it to the learner via `obs[T]` (R2-11). Block the worker at startup until weights arrive (R2-17). Log policy lag (R2-18).
10. SubprocVecEnv: forward exceptions, use shared-memory buffers, run a torch-free child, and alternate halves to overlap stepping and inference (R2-16).
11. Longer term: an ObsSpec with dict or native-dtype observations and per-agent obs transforms (R2-20); an optional inference device or a per-machine inference server (R2-21); a single definition of the step budget (R2-22).
