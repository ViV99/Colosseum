<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/task-FX1-report.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# Task FX1 report: training workers send match results in batches

**Status: DONE**

## Commits (pushed to `origin/sp4-selection`)
- `862f191` fix: workers send match results in batches, so the results queue stops dropping them on fast games
- `91ae910` docs: FX1 check: no lost match results after the batching fix, 101.8 % / 99.9 % of main's env steps/s

## What was implemented
- **`src/colosseum/worker/rollout_worker.py`**
  - Added `RESULTS_BATCH_SEC = 0.1` and `RESULTS_BATCH_MAX = 256`. The module comment gives the reason: the pass-cadence drain with a 64 KiB pipe, and one feeder wake-up (GIL) per batch instead of per match.
  - Added `ResultBatcher(q, worker_id, *, max_results, max_age_sec, clock=time.monotonic)` with these members:
    - `add(result)` puts the batch when `max_results` results are pending.
    - `flush_if_due(now=None)` puts the pending results once `max_age_sec` has passed since the last put. `now` is an optional reading of the same clock.
    - `flush()` puts whatever is pending.
    - `dropped` counts lost results.
  - A full queue drops the whole batch and counts every result in it. The first drop logs a WARNING (`Worker N: the results queue is full; dropped K match results (counted in dropped_match_results ...)`); later drops are only counted. The batcher never blocks (`put_nowait`).
  - Wiring in `rollout_worker_process`:
    - `LoopIO.report_result = batcher.add`.
    - `should_stop` calls `batcher.flush_if_due(now)` once per loop step, so a due batch goes out even when no match finishes.
    - `worker_stats()` = `loop.stats` plus `batcher.dropped` in `dropped_results`. It is used by the periodic stats report, the final stats report and the `Worker N: finished.` log line, which `test_sp4_journal_runs` parses.
    - In `finally`, `batcher.flush()` runs before the final stats report and before `flush_queue(results_queue)`.
- **`src/colosseum/launcher.py`**
  - `_RESULTS_QUEUE_SIZE = 1_000` batches. The comment gives the reason and says this is more capacity than the spec's 10 000 single results (at most 256 000).
  - New `_flatten_result_batches(items)`: flattens in arrival order. An item that is not a `list`, or a list element that is not a `MatchResult`, raises `TypeError` before any result is fed. The message names the item index, its type and a `reprlib` excerpt. There is no dual path for bare results.
  - `_drain_results` uses it.
  - The pulse queue (`_PULSE_RESULTS_QUEUE_SIZE`, its own messages with `.results`) is unchanged; the pulse tests in the full suite pass. `distributed.py` passes no results queue, and the tournament engine has its own queue with blocking puts; neither is affected.
- **`src/colosseum/worker/rollout_loop.py`**: comment only. A `LoopIO.report_result` that returns None may also be a caller that counts its own drops (the `ResultBatcher`).
- **Tests**
  - New `tests/unit/test_sp4_result_batcher.py` (injected clock):
    - the constants;
    - a batch is put at 256 pending results;
    - a batch is put after 0.1 s with no new result, and nothing is put when nothing is pending;
    - the age counts from the last put, and a caller's `now` can be passed;
    - `flush` sends the remainder in order;
    - a full queue drops whole batches, counts every result and warns once with the message's content.
  - `tests/contract/test_sp4_result_flow.py`:
    - The flatten test now puts 3+1 batches and then a later batch. It checks arrival order, and exactly-once delivery to the coordinator, the hub and the journal writer (new `_WriterRecorder`), plus the `env_steps` stamps and `_results_received`.
    - New parametrized malformed-item test: a bare `MatchResult`, a list with a dict, a tuple. Each fails loudly, and nothing is fed.
    - New worker-level due-flush test: the size cap is unreachable, and batches still arrive while the worker runs.
    - The full-queue test was rewritten. Batches of 3 with no time flush: the first batch fills the queue, and the final report must count the drops of the last flush on exit, `dropped == episodes − 3` with `episodes % 3 != 0`. It also checks the WARNING once and the `finished.` log line.
  - New `tests/integration/test_sp4_result_burst.py`, the burst test (corrected in fix round 1):
    - Two spawned producers (≤ 2 children) each put 5 000 results at once on their own `mp.Queue(maxsize=_RESULTS_QUEUE_SIZE)`.
    - The launcher's `QueueReader` drains both queues every 0.2 s, as the monitor loop does.
    - The **control** makes per-item puts against the same 1 000-item cap and must drop. Measured: it drops 4 000 every time. It does not reproduce the pre-FX1 failure: its drops come from an instant 5 000-put burst against a 1 000-slot semaphore, not from the drain cadence and the 64 KiB pipe, and against the old 10 000 cap the same burst does not overflow. So the test shows that batching keeps the queued item count far below the cap. The evidence for the mechanism is the drops diagnostic and the FX1 check (`docs/benchmarks.md` «После SP4»).
    - The batched producer must drop nothing and keep the order. Measured: 20 batches, 0 dropped.
    - Once both producers have reported, every drop is already decided, so the rest is read without the 0.2 s delay. That keeps the test at about 1.35 s (6 repeats: 1.34–1.37 s).
  - Updated to lists: `tests/unit/test_sp2_launcher_lifecycle.py` (it puts a batch), `tests/unit/test_rollout_worker_v2.py` (it expects batches), `tests/integration/test_sp3_fixed_players_worker.py` (it collects results from batches).
- **`docs/benchmarks.md`** «После SP4»: the drop paragraph was replaced with:
  - the diagnostic's numbers: main drops too, 79 % / 27–31 %; branch 73 % / 7–15 %;
  - the cause: the pass-cadence drain with a 64 KiB pipe; the feeder buffers counted by the semaphore;
  - why drain-only fixes cost 11–14 %;
  - the fix;
  - the prototype numbers;
  - then (second commit) the check table below.

  The CV section's «Журналы прорежены» sentence is unchanged, and README / CLAUDE.md / the guides are untouched.

## TDD evidence
- **RED:** `OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python -m pytest -q -rw tests/unit/test_sp4_result_batcher.py tests/contract/test_sp4_result_flow.py tests/integration/test_sp4_result_burst.py`. This is expected, because nothing existed yet:
  ```
  E   ImportError: cannot import name 'RESULTS_BATCH_MAX' from 'colosseum.worker.rollout_worker'
  E   ImportError: cannot import name 'ResultBatcher' from 'colosseum.worker.rollout_worker'   (contract, burst)
  !!! Interrupted: 3 errors during collection
  ```
  (`drops-diag/fx1/red1.txt`)
- **Mutation REDs.** Each broken variant was run against the new tests, then restored:
  - A, no `batcher.flush()` on exit: `assert 15 == (20 - 3)` (contract) and `assert ([])` (test_rollout_worker_v2). 2 failed.
  - B, final stats report before the flush: `assert 15 == (20 - 3)`. 1 failed.
  - C, no `flush_if_due` in `should_stop`: `assert (1 >= 2)` (due-flush worker test). 1 failed.
  - D, `worker_stats` without the batcher's drops: `assert 0 == (20 - 3)`. 1 failed.
  - E, launcher without flattening: 5 failed (contract and lifecycle).
  - Per-item puts in the burst scenario: the in-test control shows 4 000 of 5 000 dropped every run against the same 1 000-item cap. (Corrected in fix round 1: this is not the pre-FX1 path, whose cap was 10 000; see item 6 above.)
- **GREEN:** the focused set (the files above plus lifecycle, rollout_worker_v2, fixed_players_worker): `51 passed in 20.33s`.

## Full suite + ruff
- `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`: **2108 passed, 35 deselected in 467.95s, 0 warnings** (`drops-diag/fx1/suite.txt`).
- `ruff check .`: All checks passed.

## Measurement (after `862f191`; the diagnostic's driver, cwd = `drops-diag`)
- **Method**
  - Each run used `PYTHONPATH=<tree>/src`, and `colosseum.__file__` was verified per run: branch → `/home/viv/dev/repos/Colosseum/src`, main → `/home/viv/dev/repos/Colosseum-wt-main/src`.
  - The driver waited for the 1-minute load to be ≤ 2.0 before each run.
  - *lost* = 1 − received / finished over the whole run. Env steps/s are counters in the window: bench 15–74 s; tic-tac-toe about 5–25 s.
- **Location:** scripts `fx1/run_fx1_set.sh` and `fx1/run_fx1_ttt.sh`; outputs `fx1/out/*.json|log`. Checkpoints were deleted afterwards.

| workload | run | load before | finished | received | lost | env steps/s | queue max (batches) |
|---|---|---|---|---|---|---|---|
| bench, 4 workers, 60 s | main-1 | 1.23 | 151 147 | 109 037 | 27.9 % | 13 204 | 1000 (items) |
| | branch-1 | 1.91 | 166 626 | 166 626 | 0.0 % | 13 651 | 26 |
| | main-2 | 1.99 | 166 372 | 115 764 | 30.4 % | 13 319 | 1000 (items) |
| | branch-2 | 1.93 | 161 247 | 161 247 | 0.0 % | 13 353 | 23 |
| tic_tac_toe.yaml, 2 workers, whole run | branch-1 | 1.95 | 86 096 | 86 096 | 0.0 % | 24 123 | 8 |
| | main | 1.72 | 82 228 | 21 380 | 74.0 % | 23 749 | 1000 (items) |
| | branch-2 | 1.97 | 87 118 | 87 118 | 0.0 % | 23 339 | 8 |

- **Branch / main, by counters:**
  - bench, 4 workers: mean 13 502 / 13 261.5 = **101.8 %**;
  - tic-tac-toe config: mean 23 731 / 23 749 = **99.9 %**.
- **Drops:** 0 on every branch run.
- **Against the prototype:** these match it (0 drops; 102.5 % / 99.4 %), so the status is DONE.

## Deviations and notes
- **Extra measurement.** I added a same-session main vs branch pair on the tic-tac-toe config, which the brief did not ask for. The first branch run's 24 123 env steps/s was below the diagnostic's main number from another session (26 742). Main measured 23 749 today, so the 26 742 was a session difference, not a cost of the fix. No tuning was done.
- **API shape:**
  - `ResultBatcher` takes an injectable `clock` (constructor), and `flush_if_due(now=None)` optionally takes the caller's reading, which `should_stop` passes.
  - `add()` checks only the size cap. The time check runs once per loop step in `should_stop`; the prototype also checked time in `report_result`. Either way the delay is at most one loop step past 0.1 s.
  - The age counts from the last put, as in the prototype.
- **Contract A3 kept.** `LoopIO.report_result` returning False still counts in `RolloutLoop.stats["dropped_results"]`; the generic loop contract and its test are unchanged. The worker's batched path returns None and merges `batcher.dropped` into the stats it reports.
- **Burst test design:**
  - It holds a per-result control producer, so the test proves the scenario overflows the old path. It fails if the queue is ever made large enough that 5 000 results no longer overflow it per item; the assertion message says to raise N.
  - After both producers have reported, it drains without the 0.2 s delay, which keeps it fast.
- **Spec and plan not edited.** The spec (block 1, "10 000") and the plan's A3 (`_RESULTS_QUEUE_SIZE = 10_000`) are frozen history. The new value and its reason live in the code comment and in benchmarks.md.
- **Observation, no action taken.** In the burst probe, the reader got about one 256-result batch per 0.2 s pass when it read with the pass delay. The producer's feeder pickles the next batch while the pipe is momentarily empty, and the reader stops at that `Empty`. So drain capacity per pass is still tied to the cadence, at about one batch per feeder per pass. In real runs this was enough: at most 26 batches queued at 4 workers, 8 on tic-tac-toe. The optional 10 ms idle drain from the diagnostic would add headroom beyond 4 workers; per the ruling it was not added.

## Self-review
- The diff covers every brief item:
  - batching unit;
  - due flush on the `should_stop` path;
  - stop flush before the final stats;
  - lists only in the launcher, with a loud failure;
  - every producer and consumer of the results queue updated;
  - the tests listed in the brief;
  - docs;
  - measurement.
- No speculative features. `rollout_loop.py` has a comment-only change.
- The tests check behavior: order, counts, warnings once, exactly-once delivery. The mutation runs show that each piece of wiring is covered.

## Files changed
- `src/colosseum/worker/rollout_worker.py`, `src/colosseum/launcher.py`, `src/colosseum/worker/rollout_loop.py`
- `tests/unit/test_sp4_result_batcher.py` (new), `tests/integration/test_sp4_result_burst.py` (new), `tests/contract/test_sp4_result_flow.py`, `tests/unit/test_sp2_launcher_lifecycle.py`, `tests/unit/test_rollout_worker_v2.py`, `tests/integration/test_sp3_fixed_players_worker.py`
- `docs/benchmarks.md`

## Fix round 1 (review 1)
Commit `d479f90` fix: due result batches also go out while a chunk waits for a full trajectory queue; the burst test says what it shows (FX1 review 1). Pushed to `origin/sp4-selection`.

1. **(Important) The burst test's claim.**
   - Rewrote `tests/integration/test_sp4_result_burst.py`:
     - The docstring now says the test shows that batching keeps the queue's item count far below its cap: 5 000 results at once, no drop, order kept. Per-item puts of the same burst against the same 1 000-item cap overflow.
     - It says explicitly what the test does not show, the pre-FX1 mechanism (sustained production, a once-per-pass drain, the 64 KiB pipe, the old 10 000 cap), and points to `docs/benchmarks.md` «После SP4» for the diagnostic and the FX1 check.
   - The control is now `per_item` (formerly `one_by_one`). The test is renamed `test_a_burst_above_the_queue_cap_is_dropped_per_item_but_not_in_batches`, and the N comment and the assertion message no longer mention pipe-fulls or the old path.
   - Corrected item 6 and the mutation-RED line above. The commit message `862f191` is history and was left unchanged.
2. **(Minor) The due flush while a step stalls.**
   - `send_chunk`'s `except queue.Full` retry branch now calls `batcher.flush_if_due()` when there is a results queue. The batcher is now created before `send_chunk`, for readability.
   - The comment at the constants now gives the real delay: up to 0.1 s plus one loop step, or one 0.5 s retry while a chunk waits. The `ResultBatcher` docstring names both call sites.
   - Needed for a deterministic test: `flush_if_due()` no longer takes `now`. `should_stop` used to pass a `time.monotonic()` reading, which bypassed an injected clock. The batcher now reads only its own clock (the cost is one extra `monotonic()` call per step), and the unit test `test_the_age_counts_from_the_last_put` advances the test clock instead.
   - New contract test `test_a_due_batch_is_put_while_a_chunk_waits_for_a_full_trajectory_queue`:
     - A one-slot trajectory queue that nobody reads counts the puts that gave up on Full.
     - The batcher runs on a test clock with an unreachable size cap.
     - Once the worker is known to be retrying (≥ 1 Full) and nothing has been put, the test moves the clock. A batch must then arrive while the worker is still alive and not stopped, with `w0_e0_ep0..` in order, and the exit flush must have nothing left.
     - RED before the fix: `assert (False)` (`delivered_while_retrying`), 5.78 s (`drops-diag/fx1/red-fix1.txt`).
     - GREEN: 1.5 s. 5 repeats of the two worker-level due tests all passed.
- **Focused tests:** `OMP_NUM_THREADS=1 env -u PYTHONPATH .venv/bin/python -m pytest -q -rw tests/unit/test_sp4_result_batcher.py tests/contract/test_sp4_result_flow.py tests/integration/test_sp4_result_burst.py tests/unit/test_rollout_worker_v2.py`: **21 passed in 4.49s**. Also `test_sp3_fixed_players_worker.py` + `test_sp2_launcher_lifecycle.py`: 31 passed. Ruff: all checks passed. Per the instruction, the full suite was not run.
