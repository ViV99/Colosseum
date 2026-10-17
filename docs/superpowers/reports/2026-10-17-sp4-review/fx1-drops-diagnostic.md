<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/drops-diag/report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# Dropped match results: diagnostic (SP4 branch vs main)

Saved by the controller from the diagnostic agent's final message: the harness refused the agent's own write.

Trees: branch `sp4-selection` at 371897b; main at 1366bec (`/home/viv/dev/repos/Colosseum-wt-main`, untouched). No tracked file was edited. Machine time was about 45 min wall, most of it waiting for the 1-minute load to fall to 2.0 or below.

## Summary
1. Main drops as many results as the branch, or more, measured the same way: 27 % and 31 % at 4 workers on the bench (branch 6.8 %), and 79 % on the default tic-tac-toe config with 2 workers (branch 73 %).
2. Branch at 1 and 2 workers on the bench: 0 drops, but the backlog already reaches 2 431 and 3 828 results.
3. The main process is not the limit: it sleeps 88–96 % of each pass and uses 0.06–0.15 of a core. Per-result work is about 45 µs, refit and writer threads use almost no CPU, and nothing grows over a 100 s run.
4. Root cause: the results queue is read only in one 6–11 ms burst per 0.2 s pass. `QueueReader._read` stops at the first empty `get_nowait`, and the 64 KiB pipe holds about 100–140 results.
5. Between bursts the pipe is full (median 59.5 KB). Results pile up unpickled in the workers' feeder buffers, which the queue's semaphore counts, so `Full` hits at 10 000 (branch) or 1 000 (main). Drain capacity is about 180–420 results per pass.
6. The "late onset" is just the time to fill that backlog. In predator_prey, production also rises as episodes shorten from 21 to 12 steps.
7. Draining more often or a 1 MiB pipe stops the drops but costs 11–14 % of env steps/s (86 % of main). Once the pipe flows, every `put_nowait` wakes the worker's feeder thread, which takes the worker's GIL once per match.
8. Smallest fix that passes both criteria: workers put results in batches (one list per 0.1 s or 256 results) and the launcher flattens them. Measured: 0 drops on both workloads, 13 138 env steps/s at 4 workers (102 % of main's mean 12 823), 99 % on the tic-tac-toe config.
9. Drops stay counted per result and workers never block, but the queue size now counts batches, and the spec wording plus about four tests need updating.
10. Draining every 10 ms while idle, on top of batching, adds headroom: measured 0 drops and no throughput loss.

## 1. Method
**Driver.** `drops-diag/driver.py` runs one training in-process with the tree's own code (`PYTHONPATH=<tree>/src`; `colosseum.__file__` is checked on every run). Two workloads:
- **Bench:** the tree's own `_make_config(N)`, 15 s warm-up, 60 s measured, stopped by a timer.
- **Default config:** `configs/examples/tic_tac_toe.yaml` through `run_training`: 2 workers × 16 envs, CUDA learner (`device: auto`), 600k-step budget. One run used `--set training.total_timesteps=2000000`.

**Hooks.** Class-level wrappers installed only in the main process, valid on both trees:
- every monitor-loop step;
- the coordinator's, ratings' and PFSP's result calls, and the hub's;
- the journal writer's `submit` and `_write`, and `QueueReader.drain` and `_read`, with thread CPU;
- the launcher's sleep;
- the hub's worker-stats callback, which records each worker's `episodes` count;
- a sampler thread every 0.25 s on `time.monotonic()`: queue size, pipe fill (FIONREAD) and pipe capacity.

**Measure.**
- *finished* = sum of the workers' `episodes` stats. Each match is counted once, before it is offered to the queue.
- *received* = calls of `Coordinator.report_match_result`, including the shutdown drain.
- *lost* = 1 − received / finished.
- *window drop share* = 1 − Δ(received + qsize) / Δfinished.
- **Cross-checks:** on the branch, finished − received equals the workers' `dropped_results` exactly, and the rows in `matches.db` equal received (139 738, 23 881, 163 925).

**Load.** Before each run the driver waited until the 1-minute load was ≤ 2.0. Values near 1.9 are the decaying average left by the previous run, not live load.

## 2. Numbers
| run | load before | finished | received | lost (whole run) | window share | env steps/s | backlog max | first full (s after 1st env step) | pipe median | result bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| main, tic-tac-toe config, 2 workers | 0.31 | 82 301 | 16 982 | **79.4 %** | 79.4 % | 26 742 | 1 000 | 1.0 | n/a | 462 |
| branch, tic-tac-toe config, 2 workers | 0.44 | 88 246 | 23 881 | **72.9 %** | 84.4 % | 26 198 | 10 000 | 3.5 | n/a | ≈608 |
| main, bench, 4 workers (run 1) | 1.86 | 147 683 | 107 767 | **27.0 %** | 26.7 % | 12 669 | 1 000 | 5.5 | n/a | 462 |
| main, bench, 4 workers (run 2) | 1.94 | 156 344 | 107 757 | **31.1 %** | 33.2 % | 12 976 | 1 000 | 1.3 | 60 032 | 462 |
| branch, bench, 4 workers | 0.57 | 149 977 | 139 738 | **6.8 %** | 8.7 % | 12 553 | 10 000 | 49.4 | n/a | 608 |
| branch, bench, 2 workers | 1.94 | 77 471 | 77 471 | **0.0 %** | 0.2 % | 6 632 | 3 828 | – | 59 516 | 608 |
| branch, bench, 1 worker | 1.86 | 60 344 | 60 344 | **0.0 %** | 0.0 % | 5 065 | 2 431 | – | 59 518 | 609 |

- At 4 workers on the bench, production (about 2 040/s) is barely above the drain capacity, so the drop share depends on how long the run continues after the backlog fills. That explains 15 % in T8.2b against 6.8 % here.
- The CV tic-tac-toe result (74 %) is reproduced here (72.9 %).
- Main drops more only because its queue is 10× smaller. The drain mechanism is the same on both trees.

## 3. Branch main loop, time per pass
| step | bench, 4 workers (ms/pass) | tic-tac-toe config, 2 workers (ms/pass) |
|---|---|---|
| sleep 0.2 | 190.8 (87.8 %) | 200.2 (96.2 %) |
| `_drain_results` | 19.7 (9.1 %) | 4.9 (2.4 %) |
| · reader drain | 11.0 | 2.9 |
| · coordinator (ELO 1.5, PFSP 0.8) | 3.7 | 0.8 |
| · hub | 3.9 | 1.0 |
| · journal submit | 0.3 | 0.08 |
| `_drain_metrics` | 4.5 (2.1 %) | 2.2 (1.1 %) |
| checkpoints | 0.1 | 0.5 |
| `maybe_tick` | 0.3 | < 0.1 |
| retention / pulse / queue_depths / match-log stats / live snapshot | < 0.01 each | < 0.01 each |
| rest | 1.9 | 0.1 |
| pass length | mean 217, max 328 | mean 208, max 285 |

**Per-result costs:** unpickling 25 µs; coordinator 8.9 µs; hub 9.6 µs; journal writer thread 42 µs (runs in parallel).

**CPU use:** the main process uses 0.15 cores in total; the live-refit thread used under 0.05 cpu-s for the whole run.

**Reads per pass:** each drain reads about 400 results (bench) or about 177 (tic-tac-toe config). A read was still running at the 0.05 s join in only 17 of 349 passes (bench) and 5 of 135 (tic-tac-toe config).

## 4. Root cause
1. A worker's `put_nowait` only takes the queue's semaphore and appends to the worker's own feeder buffer. The feeder thread pickles each item and writes it into the shared pipe while holding the cross-process write lock. When the pipe is full, it blocks.
2. The main process reads once per pass, until the first `queue.Empty`, which comes as soon as the pipe is momentarily empty. Nothing reads during the following ≈200 ms of sleep.
3. The pipe is 65 536 bytes (checked with F_GETPIPE_SZ). Each burst gets the pipe's contents plus what the blocked feeders refill in a few milliseconds. Each feeder must win its own worker's GIL first, which can take up to the 5 ms switch interval.
4. For the rest of the pass the pipe stays full (median 59.5 KB at 1, 2 and 4 workers). The backlog builds in the feeder buffers and `qsize()` climbs to the cap. At the cap all workers get `Full` at once, because they share one semaphore.

**Ruled out:** per-result cost (9–10 µs per call); GIL contention (the process uses ≤ 0.15 cores); other loop steps (≤ 7 ms per pass together); the 0.05 s join (a read is seldom still running there); unpickling (25 µs per item); a cost that grows (in a 2M-step, 100 s prototype run the main loop's busy share stayed at 15–17 % per 10 s, through snapshot opponents (refresh every 30 s) and a refit at 60 s).

**Confirmed by changing only the drain:** draining every 10 ms while idle: 0 drops, pipe median 3–12 KB; a 1 MiB pipe: 0 drops, pipe median 265 KB.

## 5. Fixes measured
| prototype | tic-tac-toe config, 2 workers: lost / env steps/s (vs main 26 742) | bench, 4 workers: lost / env steps/s (vs main mean 12 823) | backlog max |
|---|---|---|---|
| branch, no fix | 72.9 % / 26 198 (98 %) | 6.8 % / 12 553 (97.9 %) | 10 000 |
| idle drain, 10 ms slices (`proto-idle`) | 0 % / 23 589 (88 %); 2M-step run: 0 % | 0 % / 11 152 and 11 018 (86.5 %) | 46 / 199 |
| 1 MiB pipe (`proto-pipe`) | 0 % / 23 863 (89 %) | not run | 997 |
| worker batching + idle drain (`proto-batch`) | 0 % / 25 643 (95.9 %) | 0 % / 13 081 (102.0 %) | 2 / 4 batches |
| **worker batching only (`proto-batchonly`)** | **0 % / 26 582 (99.4 %)** | **0 % / 13 138 (102.5 %)** | 12 / 24 batches |

- **Why the drain-only fixes fail:** once the pipe flows, every `put_nowait` wakes the worker's feeder thread. That is about 500 forced GIL hand-offs per second per worker. A full pipe hides this cost: dropped results are never pickled, on main too.
- **Noise:** most rows are single runs. Main's two bench runs differ by 2.4 %. The tic-tac-toe windows are only 18 s with a CUDA learner, so they are noisier.

**Proposed diff** (sketch against the branch's `src/`; the measured version is `drops-diag/proto-batchonly.diff`):
```diff
--- a/src/colosseum/worker/rollout_worker.py
+RESULTS_BATCH_SEC = 0.1     # one put per batch: the feeder takes this process's GIL per batch, not per match
+RESULTS_BATCH_MAX = 256
+class ResultBatcher:        # add() -> flush_if_due(); flush(): put_nowait(list); on Full: dropped += len(batch), first WARNING
@@ rollout_worker_process
-    def report_result(result) -> bool: results_queue.put_nowait(result) ... return False
+    batcher = ResultBatcher(results_queue, worker_id) if results_queue is not None else None
+    def report_result(result) -> None: batcher.add(result)          # buffered; drops counted by the batcher
+    def worker_stats(): stats = dict(loop.stats); stats["dropped_results"] += batcher.dropped; return stats
@@ should_stop:  batcher.flush_if_due() before the stats check; report_worker_stats(..., worker_stats())
@@ finally:      batcher.flush() before the final stats report and before flush_queue(results_queue)
--- a/src/colosseum/launcher.py
-_RESULTS_QUEUE_SIZE = 10_000
+_RESULTS_QUEUE_SIZE = 1_000   # batches (<= 256 results each): >= 25 s backlog at 4 workers, bounded memory
@@ _drain_results
-        results = self._drain(self._results_queue, "results", workers)
+        results = [r for b in self._drain(self._results_queue, "results", workers) for r in b]
```

**Optional headroom: idle drain on top** (`drops-diag/proto-idle.diff`): in place of `time.sleep(0.2)`, the loop sleeps in 10 ms slices and calls `_drain_results()` after each. With batching alone the pipe is still full in the median sample (about 60 KB), so the drain is still tied to the pass cadence; it keeps up with backlog ≤ 24 batches, but the margin beyond 4 workers or about 4 000 results/s is unmeasured. With it: pipe median 0, backlog ≤ 4 batches, 0.19 cores in the main process, no measurable throughput change at 4 workers.

## 6. Risks
- **Spec and test changes:** the queue size now counts batches; a drop loses a whole batch, still counted per result; workers still never block; spec block 1's "one MatchResult per finished match" needs the batch wording.
- **Lost results:** a killed worker (SIGKILL, OOM) can lose up to 0.1 s of results uncounted. Today the items in its feeder buffer are lost the same way.
- **Delay:** results, PFSP and ELO updates arrive ≤ 0.1 s later; the `env_steps` stamp shifts by the same amount.
- **Tests to update:** `tests/unit/test_rollout_worker_v2.py`; `tests/integration/test_sp3_fixed_players_worker.py`; `tests/contract/test_sp4_result_flow.py`, including the test that every drop is counted, now per batch; the launcher tests that feed single results through a `queue.Queue` (`test_sp2_launcher_lifecycle.py`) — either they put lists, or `_drain_results` keeps accepting a bare `MatchResult` (the prototype does this with an `isinstance` check).
- **Unaffected:** the pulse results queue is already batched; `distributed.py` passes no results queue.
- **Same cap on the metrics queue:** the learner puts with `put_nowait`, maxsize 100, drained once per pass. These runs carried only 53–107 items/s, but a GPU learner with `log_interval: 1` and large metrics dicts could hit it silently. Not measured further.
- **Coverage:** only tic-tac-toe with 1–4 workers was measured, plus the CV evidence for predator_prey. Games with larger results and more seats were not.

## Files (in this directory)
- Scripts: `driver.py`, `analyze.py`, `table.py`, `run_bench_set.sh`, `run_proto_set.sh`, `run_batch_set.sh`.
- Outputs: `out/*.json` and `out/*.log`, `out/table.md` (all rows), and `runs/*/` (with `metrics.jsonl` and `matches.db`; checkpoints deleted).
- Prototypes: `proto-idle/`, `proto-pipe/`, `proto-batch/`, `proto-batchonly/`, and their `*.diff` files.
