# R6: Empirical verification of Colosseum (by running it)

**Environment.** Python 3.12, torch 2.14.1+cpu, numpy 2.5.3, gymnasium 1.4.0, pydantic 2.13.5. Machine: 8 cores, 11 GB RAM, shared with other agents. Load average swung between 2 and 39 during the session, so treat absolute timings as noisy. A/B comparisons were run back to back.

**Paths.**
- `$S` = `/tmp/claude-1000/-home-viv-dev-repos-Colosseum/c1c58436-6bd4-4d8e-a7d7-ac184cbf43cf/scratchpad`
- `$W` = `$S/r6_runs`. All artifacts and logs are here.

**Common environment for every run.** `PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled PYTHONPATH=/home/viv/dev/repos/Colosseum`.

**Working directory.** Runs used `cd $W/cwd`, which holds symlinks `configs` and `examples` into the repo. This is needed because tests and configs use the relative paths `configs/...` and `./checkpoints`.

**Scripts I wrote** (all outside the repo, in `$W/scripts/`):
- `ttt_eval.py`: tic-tac-toe vs a uniformly random legal-move player or vs another checkpoint. Plays both seats and reports illegal-move losses. Has `--mask` and `--greedy` options.
- `chase_eval.py`
- `gen_ttt_expert.py`: heuristic expert (win > block > center > corner) for BC data.
- `train_logged.py`: calls `run_training`, but configures logging at import time so spawned learners and workers actually log.
- `tput.sh` and `tput_parse.py`
- Probe envs in `$W/nplayer/r6probe/envs.py`: `Solo` (1 player), `FFA3` (3 players), and `TTTActive`, which is tic-tac-toe plus `info["active"]` and `info["action_mask"]`.

**Repo hygiene.** I did not modify any repo file. My first full pytest run (from the repo root) created an empty `/home/viv/dev/repos/Colosseum/checkpoints/` through `test_full_pipeline`. I removed it; it was empty and created by my run. A different, non-empty `/home/viv/dev/repos/Colosseum/checkpoints/` (agent_0, agent_alpha and agent_beta, created 15:36–15:37) now exists. It is **not mine**: my per-file and standalone runs wrote to `$W/cwd/checkpoints` and `$W/sa_cwd/checkpoints`. Another agent probably ran tests from the repo root, so I left it in place.

Status tags used below: **[VERIFIED-BY-RUNNING]** means I observed it directly. **[CODE-READ]** means it is inferred from source only.

---

## 1. Test suite results

### Full single-process run [VERIFIED-BY-RUNNING]

Command:
```
pytest tests/ -v --durations=25 -p no:cacheprovider
```
Log: `$W/pytest_full_hung.log`

- **The run HUNG.**
  - 86 tests had passed when `tests/test_integration.py::test_full_pipeline` stalled.
  - It sat for more than 10 minutes with 0% CPU until I killed it.
  - The forked children were stuck: one was single-threaded and blocked in `futex_wait_queue_me`, the other in `poll`.
  - Before that, `test_integration.py::test_worker_produces_chunks` had FAILED. The forked worker produced no chunk within 10 s, right after the gRPC tests had started background threads in the pytest process.

### Per-file runs [VERIFIED-BY-RUNNING]

Each file was run in a fresh process with `timeout 420`, `-X faulthandler`, `--rootdir=<repo>`, and `--basetemp` under `$W`. Script: `$W/scripts/run_perfile.sh`. Logs: `$W/perfile/*.log`.

| file | result |
|---|---|
| test_action_masking | 12 passed |
| test_appo | 7 passed |
| test_bc | 7 passed |
| test_composite_actions | 26 passed |
| test_config | 12 passed |
| test_distributed | 3 passed |
| test_distributions | 6 passed |
| test_eval | 6 passed |
| test_grpc | 5 passed |
| **test_integration** | 2 passed, then **`test_full_pipeline` HANGS (timeout 420 s)** |
| test_milestone2 | 5 passed (177 s) |
| test_multi_agent | 10 passed (184 s), 1 warning: *"This process is multi-threaded, use of fork() may lead to deadlocks"* |
| test_performance | 6 passed (the `torch.compile` tests ran; none were skipped) |
| test_ratings | 11 passed |
| test_recurrent | 21 passed |
| test_review_fixes | 12 passed, 2 errors |
| test_subproc_vec_env | 7 passed |
| test_vec_env | 5 passed |
| test_vtrace | 8 passed |

**Totals.**
- 174 tests collected, versus the documented "148 passing, 2 skipped". The docs are stale.
- 173 pass when each file is isolated.
- **0 skipped.**
- 1 test, `test_full_pipeline`, hangs whenever it runs after other tests in the same process.

**Errors in test_review_fixes.** The 2 errors (`test_resume_from_path`, `test_resume_from_none_returns_none`) came from my harness: the `--basetemp` parent directory disappeared mid-sequence. Re-run with a fresh basetemp, the file gives **14/14 passed**. Not a product bug.

### Root cause of the hang [VERIFIED-BY-RUNNING]

- `test_full_pipeline` **passes alone** (`-k full_pipeline`, 1 passed in 113 s).
- `-k "learner_trains or full_pipeline"` **hangs** (killed at 240 s).
- `test_learner_trains` runs a torch training step in the pytest process, which starts intra-op threads. `test_full_pipeline` then builds a `Launcher` with the default Linux start method, **fork**, which deadlocks in the child. The earlier `test_worker_produces_chunks` failure is the same problem, triggered by gRPC threads.

**Diagnosis: test-harness bug.** The production CLI calls `mp.set_start_method("spawn")`. The tests never do, and `conftest.py` doesn't either. A `tests/conftest.py` with `multiprocessing.set_start_method("spawn", force=True)` would fix it.

**Other test hygiene:** `test_full_pipeline`, `test_milestone2`, the `run_*` scripts and the example configs all write `./checkpoints` into the current directory, which is the repo root when you run from there.

### Slowest tests (isolated)

| test | time |
|---|---|
| test_multi_agent_pipeline | 181 s |
| test_full_pipeline_with_checkpoints | 176 s |
| test_full_pipeline (alone) | 113 s |
| test_evaluate_three_agents | 42 s |
| test_offline_bc_loss_decreases | 36 s |
| test_offline_bc_from_tensors | 15 s |
| test_evaluate_two_agents | 12 s |

All other tests take under 10 s. The pipeline tests are slow mostly because of torch thread oversubscription (see Section 3). No NaN or inf warnings were seen anywhere.

### Standalone scripts [VERIFIED-BY-RUNNING]

Run from `$W/sa_cwd`. Logs: `$W/standalone/`.

| script | outcome |
|---|---|
| run_worker_test.py | OK, 3 s |
| run_pipeline_test.py | Exit 0 after 107 s, prints "Pipeline completed!", **but a worker traceback appears at shutdown**: `FileNotFoundError` in `_sync_weights → torch rebuild_storage_fd` (see R6-02) |
| run_distributed_test.py | OK, 33 s. "Final published policy_version 75. Distributed gRPC pipeline OK!" |
| run_subproc_pipeline_test.py | Exit 0 after 52 s, **same worker traceback**, then "All worker processes exited unexpectedly; stopping." It still prints "completed!" |
| run_scaled_test.py | **Timed out at 600 s** (330 of 390 train steps), under load of about 30 with 8 threads per process. 19× "Checkpoint ckpt_v50 not found, using latest". 41 leaked semaphores at kill. See R6-05. |

---

## 2. Learning results

All evaluations use N=500 games: 250 per seat against a uniformly random legal-move player. The agent plays without masking unless the column says mask=True. "Illegal" counts the agent's games lost by an illegal move.

**Random-vs-random reference** (untrained net with masking, greedy): X wins 44%, O wins 18%, 31% overall.

### 2.1 Tic-tac-toe self-play, stock example [VERIFIED-BY-RUNNING]

| run | checkpoint | mode | W / D / L total | seat X W | seat O W | illegal |
|---|---|---|---|---|---|---|
| untrained | init | sample | 0.042 / 0.002 / 0.956 | 0.060 | 0.024 | 426/500 |
| run 1: default config, 200k steps, 2 workers, 8 threads per process, load ~30. **25.5 min wall, 9784 CPU-s** | v50 | sample | 0.052 / 0.010 / 0.938 | 0.076 | 0.028 | 416 |
| | v200 | sample | 0.068 / 0.010 / 0.922 | 0.088 | 0.048 | 409 |
| | v750 (final) | sample | 0.108 / 0.012 / 0.880 | 0.160 | 0.056 | 339 |
| | v750 | greedy | 0.008 / 0.054 / 0.938 | 0.004 | 0.012 | 340 |
| | v750 vs v300 | sample | 0.606 / 0.002 / 0.392 | 0.572 | 0.640 | 175 (opponent 284) |
| run 2: 1M steps, `OMP_NUM_THREADS=1`. **5.3 min wall, 596 CPU-s** | v1500 | sample | 0.478 / 0.076 / 0.446 | | | 96 |
| | v2500 | sample | 0.688 / 0.056 / 0.256 | 0.780 | 0.596 | 48 |
| | v3750 | sample | 0.754 / 0.054 / 0.192 | 0.868 | 0.640 | 23 |
| | **v3750** | **greedy** | **0.824 / 0.066 / 0.110** | 0.896 | 0.752 | **1** |
| | v3750 vs v1500 | sample | 0.712 / 0.012 / 0.276 | 0.924 | 0.500 | 16 (opponent 75) |

**Verdict: it learns, but slowly.** The stock example needs about 1M env steps (3900 updates) to beat random reliably. At 200k steps (the default budget is 100k) it still loses 68% of games to illegal moves.

Two reasons:
- **The example env gives no `info["active"]` and no `action_mask`.** The worker therefore records the inactive player's ignored move as a real transition. Those transitions get rewards caused by the opponent's move, for example +1 when the opponent plays illegally. The observation also cannot tell whose turn it is: an equal-mark board is "my turn" for X and "not my turn" for O. Logged loss is noisy (0.20 to 0.28 for most of run 2).
- **Thread oversubscription** makes the same 200k steps take 25 minutes (Section 3).

### 2.2 Same game with `info["active"]` and `info["action_mask"]` (TTTActive) [VERIFIED-BY-RUNNING]

200k steps, `OMP_NUM_THREADS=1`: **63.6 s wall, 128 CPU-s**. Logged loss falls from 0.34 to about 0.0.

| checkpoint | mode | W / D / L | seat X W | seat O W |
|---|---|---|---|---|
| init | mask, greedy | 0.312 / 0.180 / 0.508 | 0.444 | 0.180 |
| v300 | mask, greedy | 0.678 / 0.062 / 0.260 | | |
| v750 | mask, greedy | **0.770 / 0.006 / 0.224** | 0.852 | 0.688 |
| v750 vs v300 | mask, sample | 0.566 / 0.042 / 0.392 | 0.700 | 0.432 |
| v750 | no mask, greedy | 0.000 W, 500/500 illegal | | |

The framework's turn-based and masking path learns about 5 times faster per sample than the stock example.

Note the last row: a policy trained with masks puts its highest probability on illegal cells when run without a mask. Every inference path must apply the mask. The project's `eval` does this; any external bot or export must too.

### 2.3 Multi-agent league (`tic_tac_toe_multi.yaml`) [VERIFIED-BY-RUNNING]

Overrides: 1M steps, chunk 32, batch 8, 8 envs, 2 workers, `self_play_ratio=0` (the config default), `OMP_NUM_THREADS=1`. 397 s wall.

| agent | mode | W / D / L vs random | seat X W | seat O W | illegal |
|---|---|---|---|---|---|
| alpha v3750 | greedy | 0.764 / 0.012 / 0.224 | 0.888 | 0.640 | 5 |
| beta v3250 | greedy | 0.404 / 0.138 / 0.458 | 0.516 | 0.292 | 36 |
| alpha vs beta | sample | alpha 0.698 / 0.020 / 0.282 | 0.920 | 0.476 | |

- **Both agents learned**, alpha much more.
- **The run did not finish cleanly.** Alpha's learner reached 3906 steps and exited. Both workers then crashed reading alpha's queued weights, and beta's learner crashed reading the dead workers' chunks. **Beta stopped at step 3440 of 3906** (R6-02).
- **Seat assignment is fixed** (R6-03). I checked it with the real `Coordinator` over 400 matches:
  - `self_play_ratio=0`: every match is (alpha, beta). Alpha is always X and beta always O.
  - `self_play_ratio=0.5`: 198 matches are (alpha, alpha) and 202 are (alpha, beta). Beta **never** gets solo or self-play matches and never plays X.
- No ELO or win-rate summary is printed or logged anywhere when WandB is off. The coordinator computes ratings but nothing outputs them.

### 2.4 Chase (composite Dict action, `chase.yaml`) [VERIFIED-BY-RUNNING]

Overrides: 40k steps, 1 worker, 8 threads per process. Killed by timeout at 900 s (about step 200 of 250).

| checkpoint | W vs random (sample) | mean final distance to target |
|---|---|---|
| init | 0.455 | 5.35 |
| v50 | 0.453 | 5.39 |
| v150 | 0.463 | 5.37 |
| v100 / v200 | 0.41 / 0.40 | 5.84 |

**No learning signal at this budget.** The reward is a sparse ±1 relative to an independent opponent, and the run had only about 1600 episodes. Weight drift was tiny: L2 distance 0.12 from v75 to v175. Inconclusive about correctness; this needs a 10× longer run.

Side findings:
- `ChaseEnv.step` crashes with `TypeError` on actions from its own `action_space.sample()` under numpy 2.5. It calls `float()` on a shape-(1,) array.
- `ActionSpec.decode` returns a Python scalar for `Box(shape=(1,))`, not a shape-(1,) array. That is why training does not hit the crash.

### 2.5 N-player probes (`num_players` 1 and 3) [VERIFIED-BY-RUNNING]

60k steps, 1 worker, `OMP_NUM_THREADS=1`.

| probe | result | P(best action 4): init → v300 |
|---|---|---|
| Solo, 1 player (reward = pick/4) | Runs | 0.19 → **0.31** |
| FFA3, 3 players (highest unique pick wins) | Runs and finishes cleanly | 0.18 → 0.35–0.39 |

**Solo crash at the end.** The worker reached `total_timesteps` first and exited. The learner then crashed with `EOFError` while unpickling a queued chunk (R6-02), and the main process logged "All worker processes exited unexpectedly". The learner stopped at about 370 of 375 updates.

Not tested: team control, team vs team, asymmetric 1-vs-N. The env API has a single shared observation and action space for all players, so asymmetric roles are not expressible [CODE-READ].

### 2.6 Space Miners [NOT RUN]

`examples/space_miners/game_engine.py` imports `Box2D`. Box2D is not a declared dependency in `pyproject.toml` and is not installed, and I was not allowed to install it. The example is 2-player, simultaneous move, with a Dict action (Box(6) + 3×Discrete(2)) and `max_ticks=1000`. It is unusable out of the box.

### 2.7 Behavioral cloning [VERIFIED-BY-RUNNING]

Data: `gen_ttt_expert.py`, 3000 games, 23,698 samples. Command:
```
colosseum bc -c configs/examples/tic_tac_toe.yaml --data $W/bc/ttt_expert.pt --output $W/bc/bc_weights.pt --epochs 10
```
- Loss fell from about 0.9 to 0.4628 in the final epoch.
- **The CLI prints `Loss=0.6718`**, which is the mean over all epochs, not the final loss.
- It took 4 min 50 s wall and 15.6 CPU-min for about 930 tiny MLP steps (8 threads, heavy load).

| policy | mode | W / D / L vs random | illegal |
|---|---|---|---|
| BC | sample | 0.814 / 0.114 / 0.072 (X 0.912, O 0.716) | 4/500 |
| BC | greedy | **0.882 / 0.110 / 0.008** (X 0.924, O 0.840) | 0/500 |
| RL from BC (`training.resume_from=bc_weights.pt`), v100, after 26 s of training | greedy | 0.812 / 0.148 / 0.040 | 0 |
| same run, v300 | greedy | 0.794 / 0.150 / 0.056 | 0 |

BC to RL hand-off works: weights load and play starts at BC strength. A short RL phase slightly degrades greedy play (expected early on; the value head is untrained after BC).

The `OfflineBCTrainer` docstring claims `action_masks` in the data are honored. `load_data` and `add_data` ignore them [CODE-READ].

---

## 3. Throughput [VERIFIED-BY-RUNNING]

**Method.** `tput.sh` runs `train_logged.py` for 70 s, then stops it with `timeout -s INT`.
- Learner updates/s and chunks/s come from learner log timestamps, excluding the first 10 steps.
- Env steps/s = chunks/s × T / 2, because both slots collect in plain self-play.

Workers never logged their final step counts because SIGINT kills them (Section 4). Logs and `top` snapshots are in `$W/tput/`.

### Tic-tac-toe (T=32, 8 envs per worker, batch 8)

| workers | threads | learner upd/s | chunks/s | env steps/s | CPU per process (top) |
|---|---|---|---|---|---|
| 1 | default (8) | 1.93 | 15.3 | ~245 | learner 390%, worker 410% |
| 2 | default | 0.69 | 5.5 | ~88 | each process ~230–250% |
| 4 | default | **0.63** | 5.0 | ~80 | each process ~165–200% |
| 1 | `OMP_NUM_THREADS=1` | 11.12 | 89.0 | ~1420 | worker 150%, learner 50% → **worker-bound** |
| 2 | 1 | 14.70 | 117.6 | ~1880 | workers ~90–100%, learner 70% |
| 4 | 1 | **16.48** | 131.8 | ~2110 | learner 90%, workers 20–80% → **learner-bound** |

### Chase (T=20, 4 envs per worker, batch 8)

| workers | threads | learner upd/s | env steps/s |
|---|---|---|---|
| 1 | 1 | 10.43 | ~835 |
| 2 | 1 | 13.76 | ~1100 |
| 4 | 1 | 15.39 | ~1230 |
| 2 | default | **0.78** | ~61 |

**Findings.**
- **No code calls `torch.set_num_threads`** (grep-confirmed). Every worker and the learner start 8 intra-op threads on an 8-core box. With default settings throughput **gets worse as workers are added**: 1.9 → 0.7 → 0.6 upd/s. It is **7 to 26 times slower** than with one thread per process.
  - Example: the same 200k-step tic-tac-toe run took 1531 s wall and 9784 CPU-s with defaults, versus 64 s and 128 CPU-s with `OMP_NUM_THREADS=1`. External load also differed: about 30 during the default run versus about 3.
  - The pipeline tests (113–184 s) and BC suffer from the same thing.
- With one thread per process, scaling is sub-linear (1→4 workers gives about 1.5×). **It saturates at about 15–16 learner updates/s.** The learner becomes the bottleneck at 4 workers even for a 15k-parameter MLP.
  - The learner loop per step does: unpickling of shared-memory chunk tensors, a forward and backward pass, a **CPU clone of the full state_dict, pushed to every worker's queue every step** (`weight_push_interval` defaults to 1), and a checkpoint check.
  - The learner trains on a partial batch whenever the queue holds fewer than `batch_chunks` chunks. Its loop ends after a fixed number of *updates*, so `total_timesteps` is not honored. Example: 200k configured, 139k agent transitions consumed in the TTTActive run.
- With one worker, the worker is the bottleneck (150% CPU, learner at 50%).

---

## 4. Robustness probes [VERIFIED-BY-RUNNING]

Logs are in `$W/robust/`.

| probe | outcome |
|---|---|
| **Kill one worker** (`kill -9`, 2-worker TTT, 20 s in) | **The whole run dies within about 1 s.** The learner crashes with `ConnectionResetError` in `torch rebuild_storage_fd` while unpickling a chunk the dead worker had queued. The main process then logs "All learner processes exited", and the surviving worker is stopped. No fault tolerance; one worker crash kills training. |
| **Learner or worker exits normally** (end of training: multi-agent, solo, pipeline scripts) | The same crash cascade: the surviving side crashes on queued tensors from the exited process. In multi-agent runs the agent that finishes first kills the others' training. |
| **SIGTERM to the main process only** | **Main exits immediately and its children are orphaned** (reparented to init). The learner and 2 workers kept running at **62–77% CPU each, indefinitely**. I killed them manually after 51 s. There is no SIGTERM handler, so the `finally: _shutdown()` path never runs. This matters for docker and k8s, which stop containers with SIGTERM. |
| **Single SIGINT to the process group** (real Ctrl-C) | Main logs "Received interrupt, stopping... All processes stopped" and no processes are left behind. Each child prints a `KeyboardInterrupt` traceback (3 tracebacks), and workers never log their final counts. Acceptable but noisy. |
| `timeout -s INT` on the group (double SIGINT) | Same as above, plus a second `KeyboardInterrupt` raised inside the `except` handler in `launch()`. Shutdown still completes. |
| **Resume from a checkpoint id** (`training.resume_from=ckpt_v3750`, copied checkpoint directory) | Works. Logs "Resume: loaded checkpoint ckpt_v3750" and "resumed from checkpoint at step 3750". Updates continue from 3760, new checkpoints from v3800 on. The optimizer state is loaded (code path; no explicit log). Caveats: an LR schedule restarts at step 0 (scheduler not restored), and the remaining budget is `total_train_steps − 3750`. |
| **Resume from a .pt path** (BC weights) | Works, policy_version 0 (Section 2.7). |
| **Checkpoint eviction after resume** | **Data loss.** The `CheckpointManager` evicted the copied `ckpt_v3500` and `ckpt_v3750` by the absolute `path` stored in their `meta.json`. That path points to the *original* run directory (`$W/ttt_sp1t/ckpts/...`), so the original checkpoints were deleted, while the copies stayed on disk beyond `pool_size` (12 directories at pool 10/10). See R6-06. |
| **Reusing a checkpoint directory across runs** (`run_scaled_test` after other scripts in the same `./checkpoints`) | Earlier runs' checkpoints are scanned into the index and **used as self-play opponents**. A new save with the same id (`ckpt_v50`) is then deleted by FIFO eviction of the stale index entry, which has the same path. The result is 19× "Checkpoint ckpt_v50 not found, using latest". See R6-05. |
| `colosseum eval` | See Section 5. |

---

## 5. `colosseum eval` [VERIFIED-BY-RUNNING]

Command:
```
colosseum eval -c configs/examples/tic_tac_toe.yaml -a new:.../ckpt_v4250/model.pt -a old:.../ckpt_v1500/model.pt -a bc:$W/bc/bc_weights.pt -n 600
```
It ran in 2.4 s. Output:

```
        bc vs new        |     201 |   153 |    15 |   33 | 0.761 [0.881, 0.954]
        bc vs old        |     211 |   186 |    17 |    8 | 0.882 [0.875, 0.949]
       new vs bc         |     201 |    15 |   153 |   33 | 0.075 [0.046, 0.119]
       new vs old        |     188 |   139 |    41 |    8 | 0.739 [0.672, 0.797]
       old vs bc         |     211 |    17 |   186 |    8 | 0.081 [0.051, 0.125]
       old vs new        |     188 |    41 |   139 |    8 | 0.218 [0.203, 0.328]
```

- **The CI is wrong for every "reversed" row.**
  - Example: `bc vs new` WR 0.761 with CI [0.881, 0.954]; the interval does not contain the estimate.
  - Cause: the reversed row's CI is computed as `1 − CI(other's win rate)`, which is invalid when there are draws.
  - Because rows are sorted by name, the wrong rows here are bc–new and old–new.
- Each pair appears twice.
- `-n` is the **total** number of matches split across all pairs (about 200 per pair here), but the CLI help says "Matches per agent pair".
- Seats are randomized per match through `random.sample`, which is good. But there is **no seat breakdown**, and in tic-tac-toe seat X is worth roughly a 2× win-rate difference.
- Draws count as non-wins in WR. Action masks are applied. A `--deterministic` flag exists.
- Fixed-length vectorized eval slightly over-samples short episodes [CODE-READ].

---

## 6. Bugs found

### R6-01: Tests deadlock under the default fork start method [Medium, test infra] — CONFIRMED
- **Location:** `tests/conftest.py` (no `set_start_method`); `tests/test_integration.py`, `tests/test_multi_agent.py`, `tests/test_milestone2.py`.
- **Symptom:** the full suite hangs forever at `test_integration.py::test_full_pipeline`; `test_worker_produces_chunks` fails after the gRPC tests; a "multi-threaded fork()" DeprecationWarning appears.
- **Reproduce:**
  ```
  cd $W/cwd && pytest <repo>/tests/test_integration.py -k "learner_trains or full_pipeline" --rootdir=<repo>
  ```
  This hangs, while `-k full_pipeline` alone passes in 113 s.
- **Root cause:** `Launcher` forks after torch, OpenMP or gRPC threads already exist in the pytest process. Production uses spawn, so this is test-only. Fix: `mp.set_start_method("spawn", force=True)` in `conftest.py`.

### R6-02: Torch shared-memory tensors in mp.Queues: one process exiting crashes the others [High] — CONFIRMED
- **Location:** `learner.py` `_push_weights` and `_collect_chunks`; `rollout_worker.py` `_sync_weights` and `tq.put(chunk)`; standard `mp.Queue` carrying torch tensors.
- **Symptom:** `FileNotFoundError`, `ConnectionResetError` or `EOFError` in `torch/multiprocessing/reductions.py rebuild_storage_fd`.
  - Killing 1 of 2 workers kills the run.
  - In a multi-agent league, the first learner to finish kills the workers and the other learner (beta stopped at 3440 of 3906).
  - Solo run: the learner dies when the worker finishes first.
  - Tracebacks appear at the end of `run_pipeline_test` and `run_subproc_pipeline_test`.
- **Reproduce:** Section 4 "kill one worker" (`$W/robust/killworker.log`), or `train_logged.py configs/examples/tic_tac_toe_multi.yaml ... total_timesteps=1000000` (`$W/ttt_multi/train.log`).
- **Root cause:** importing torch registers `ForkingPickler` reductions. Tensors put on a queue are sent as fd handles served by the *sending* process's resource sharer. When the sender exits, any item still queued cannot be rebuilt. No `try/except` exists around get/put, and there is no fault tolerance. Fixes: send numpy or bytes (or `tensor.numpy()` copies), or catch the error and drop the item; plus per-agent completion handling.

### R6-03: League matchmaking has fixed seats and the primary agent monopolizes solo matches [High for league quality] — CONFIRMED
- **Location:** `launcher.py` (`generate_match_configs(trainable_agents[0], ...)` at launch and refresh); `matchmaker.py` `PFSPMatchmaker._arena_match` (`agents=[agent_id, opponent]`, slot `i % 2`).
- **Symptom:** in 400 sampled matches, alpha is always seat 0. Beta never plays seat 0 and never gets self-play or solo matches. PFSP only ever selects opponents for alpha.
- **Reproduce:** the Coordinator snippet from Section 2.3 (`self_play_ratio` 0.0 and 0.5).
- **Root cause:** matches are generated for one "primary" agent only, and slot order is deterministic. Fix: rotate the primary agent and shuffle slots.

### R6-04: Severe torch thread oversubscription [High, performance] — CONFIRMED
- **Location:** worker and learner process targets; no `torch.set_num_threads` anywhere.
- **Symptom:** 7 to 26 times lower throughput. Throughput *drops* as workers are added (TTT: 1.93 → 0.63 upd/s for 1 → 4 workers, versus 11.1 → 16.5 with `OMP_NUM_THREADS=1`).
- **Reproduce:** `$W/scripts/tput.sh ttt_w4_omp0 4 0 configs/examples/tic_tac_toe.yaml` versus `... omp1 4 1 ...`.
- **Root cause:** each of the N+1 processes starts 8 intra-op threads on 8 cores, and small-batch inference spins them. Fix: `torch.set_num_threads(1)` in workers (configurable) and a bounded thread count in the learner.

### R6-05: Checkpoint directory reuse mixes runs and loses fresh checkpoints [Medium] — CONFIRMED
- **Location:** `checkpoint_manager.py` (`_scan_existing` plus `save` plus FIFO eviction; the index is not re-sorted and allows duplicate ids).
- **Symptom:** a new run uses stale checkpoints from previous runs as opponents. Saving a `ckpt_vN` that already exists from an old run appends a duplicate index entry, and eviction of the old entry `rmtree`s the directory the new save just wrote. Result: 19× "Checkpoint ckpt_v50 for agent_0 not found, using latest".
- **Reproduce:** run `tests/run_pipeline_test.py`, `run_distributed_test.py`, then `run_scaled_test.py` from the same directory (`$W/standalone/run_scaled_test.log`).
- **Fix:** per-run subdirectories, dedupe by id, re-sort after append.

### R6-06: Eviction deletes by the absolute path stored in meta.json, outside the pool directory [High, data loss] — CONFIRMED
- **Location:** `checkpoint_manager.py` `save()` eviction (`shutil.rmtree(Path(oldest.path))`) and `_scan_existing` (reads `path` from meta).
- **Symptom:** after copying checkpoints into a new run directory and resuming, eviction deleted the **original** run's `ckpt_v3500` and `ckpt_v3750` in `$W/ttt_sp1t`, while the copies remained (12 directories on disk for pool 10).
- **Reproduce:** Section 4 resume probe (`$W/robust/resume.log`). Then `ls $W/ttt_sp1t/ckpts/agent_0` no longer shows v3500 or v3750.
- **Fix:** derive the path from `base_dir/agent/ckpt_id`; never trust the stored path.

### R6-07: SIGTERM to the main process orphans the learner and workers, which keep consuming CPU [High for docker/k8s] — CONFIRMED
- **Location:** `launcher.py` (handles only `KeyboardInterrupt`; no SIGTERM handler). A related problem: Ctrl-C prints a `KeyboardInterrupt` traceback in every child.
- **Reproduce:** start training with `setsid`, then `kill -TERM <main>`. The children keep running at 60–77% CPU, reparented to init.
- **Fix:** install a SIGTERM handler that raises or sets `stop_event`; set `PR_SET_PDEATHSIG` or have workers poll the parent PID; children should ignore SIGINT.

### R6-08: CLI training has no visibility into progress [Medium, usability] — CONFIRMED
- **Location:** `launcher.run_training` configures logging only in the main process. Spawned learners and workers have no handlers, and the metrics go only to WandB.
- **Symptom:** `colosseum train` prints no losses, no step counts, no worker throughput and no ELO; only checkpoint saves. `run_*` scripts show logs only because they call `basicConfig` at import. The learner also logs `device=auto` instead of the resolved device.
- **Reproduce:** `$W/ttt_sp/train.log` (only 2 lines mention Learner or Worker, both from the main process).

### R6-09: The tic-tac-toe example learns poorly because the env lacks `active` and `action_mask` [Medium, example quality] — CONFIRMED
- **Symptom:** after 200k steps (2× the default budget), 68% of games vs random are lost by illegal moves. It takes about 1M steps to reach 82% (greedy). The same game with active and mask info reaches 77% in 200k steps and 64 s.
- **Root cause:** inactive-player no-op moves are recorded as real transitions and receive opponent-caused rewards, and the observation is turn-ambiguous across seats. The framework already supports `info["active"]` and `action_mask`; the flagship example doesn't use them.

### R6-10: `colosseum eval` reversed-row confidence intervals are wrong [Medium] — CONFIRMED
- **Location:** `eval.py` `evaluate_agents`: `ci_lower_a=1 - result.ci_upper_a` and so on.
- **Symptom:** "bc vs new 0.761 [0.881, 0.954]".
- **Also:** `-n` is the total, not per pair as the help says; rows are duplicated; there is no seat breakdown.

### R6-11: Shutdown and completion semantics [Low/Medium] — CONFIRMED
- `total_timesteps` is not honored: the learner stops after N updates that may be partial batches, and each worker stops independently at `total/num_workers`.
- In the solo probe the worker finished first, which triggered R6-02. The main process logged "All worker processes exited unexpectedly" on what was a normal finish.

### R6-12: Smaller issues [Low] — CONFIRMED unless noted
- `colosseum bc` prints the mean epoch loss as "Loss=" (0.6718 versus a final 0.4628).
- `OfflineBCTrainer` claims `action_masks` support that does not exist [CODE-READ].
- `ActionSpec.decode` collapses `Box(shape=(1,))` to a scalar.
- `ChaseEnv.step` crashes with spec-conformant actions under numpy 2.5.
- Space Miners needs `Box2D`, which is undeclared in `pyproject.toml`.
- On resume the LR scheduler restarts from 0 [CODE-READ].
- 23–41 leaked-semaphore warnings at killed or timed-out shutdowns.

---

## 7. Overall verdict: does it work as documented?

**Partly.** The core APPO, V-trace, worker and learner loop is functional and **does learn**:
- Tic-tac-toe vs random reaches 82% wins (greedy) after 1M steps; the turn-aware variant reaches 77% after 200k.
- League agents improve.
- BC reaches 88%, and RL starts cleanly from BC weights.
- 1-player and 3-player games run.
- The gRPC distributed script passes.
- Resume works.

But the "148 passing, 2 skipped" claim is stale and wrong in both directions:
- The suite has 174 tests, none skipped.
- The full suite **hangs** in a single process, due to the fork deadlock in the tests.

Runtime problems contradict the documented "dynamic add/remove workers" and "works on k8s":
- One worker dying, or one learner finishing, **crashes the whole run** (R6-02).
- SIGTERM orphans processes (R6-07).
- Checkpoint management can **delete other runs' checkpoints** (R6-06, R6-05).
- League training uses fixed seats and a single primary agent (R6-03).
- Default thread settings make training 7 to 26 times slower than necessary (R6-04), so the shipped example configs barely learn within their budgets.
- `colosseum train` gives no progress output without WandB (R6-08).

**Not verified:** Space Miners (Box2D missing), real multi-machine gRPC, GPU/AMP, recurrent policies end-to-end, team or asymmetric modes.
