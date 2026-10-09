> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/task-FIX-1-report.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# FIX-1: flaky `test_resume_continues_versions_and_checks_the_role_signature`

Status: DONE. Test-only fix (the race is in the test; no product bug found).

## Root cause (reproduced)

The suspected cause ("the resumed ~1500 steps run out before the first train step") was not what happened.
The real mechanism: **the first run overshoots its 1500-step budget so far that the resume starts at or past the
second run's fixed 3000-step budget.** The second run then stops at once (rc 0) and its learner never trains, so
`train_steps(second, ...)[0]` raises IndexError.

Why the overshoot is large:
- Workers add env steps to the shared counter in batches, about every 0.5 s (`core/ipc.BatchedCounter`).
  At ~2000-3000 env steps/s that is 1000-1500 steps per flush.
- Main polls every 0.2 s, and workers keep stepping until they see `stop_event`.
- The final checkpoint's `meta.env_steps` is read after the workers' exit flush.
  This value is where the resume continues (`Launcher._resolve_resume`).

Evidence: a harness (`/tmp/fix1scripts/repro.py`, the test's two runs with the same overrides), 12 runs on an
idle machine. **3 of 12 failed**, each with first-run `env_steps` at or above 3000:
```
first: final_env_steps=3280 v=99 | second: train_recs=0 | Training budget reached: 3280 env steps (budget 3000)
first: final_env_steps=3064 v=91 | second: train_recs=0 | Training budget reached: 3064 env steps (budget 3000)
first: final_env_steps=3136 v=95 | second: train_recs=0 | Training budget reached: 3136 env steps (budget 3000)
(the other 9: first final_env_steps 2168-2896, second run trained 22-101 steps)
```
Main log of a failing pair (`/tmp/fix1_e5l0_azh`):
```
first:  Training budget reached: 2416 env steps (budget 1500)        # detection, then +864 during shutdown
second: Resume: env-step counter continues from 3280
second: Training budget reached: 3280 env steps (budget 3000)        # 8 ms after the resume line
learner: finished. Total train_steps=99                               # = resumed version, no step taken
```
The flake depends on speed, not load. Under a 12-process CPU hog the first-run overshoot was smaller (1552-2472),
because overshoot is about throughput × flush/stop latency. A fast moment in the suite is enough to trip it.

Not a product bug:
- Counting the steps actually taken (including the shutdown tail) is correct accounting.
- A resume at or past the budget stopping at once with rc 0 is the intended behavior.
- `BatchedCounter`'s ~0.5 s granularity is documented in `rollout_worker`.

## Change

`tests/integration/test_sp2_game_runs.py`:
- The resumed budget is now `final["env_steps"] + 2000`, counted from where the first run really stopped, instead
  of a fixed 3000. The same budget is used for the third run, which fails on the role signature before any step.
- New assertion: the second run's main log has `env-step counter continues from {final env_steps}`. This
  strengthens what the test proves: the env-step counter continues across resume, as well as the version.
- All existing assertions are kept: the first train step is `version + 1`, all new checkpoints are `> version`,
  and a mismatched role signature is a `Config error` naming the checkpoint path, with no traceback.

Why 2000 always leaves room for a train step: I measured how far a worker gets while its learner consumes nothing.
A `sitecustomize` replaced `learner.collect_batch` with an endless sleep, after the initial weights are pushed.
The full trajectory queue (`queue_size` 16, 16-step chunks) stopped the worker at **488 env steps**, the same in
2 of 2 runs (`chunks_sent 24`, `train records 0`). The real learner holds at most 1 chunk before it trains
(`batch_chunks` 2). The main process stops only after the counter has really advanced by the remaining budget.
So 2000 fresh steps cannot pass before at least one train step (about a 4× margin). The cost is about 1 s per run.

## Verification
- Target test, 10× in a row (idle): 10 passed (each ~12.2-12.6 s).
- Under artificial CPU load (12 busy processes on 8 cores): 4 passed (71.6 s, 36.9 s, 71.7 s, 70.8 s).
  A 5th run started after the load had ended and passed in 12.6 s.
- Full fast suite: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` → `1232 passed, 27 deselected in 348.21s`,
  0 warnings.
- `.venv/bin/ruff check .` → All checks passed.

## Concerns / notes
- Audit of other resume tests: the only other one is `tests/integration/test_sp2_league_runs.py` (resume LR
  continuation). It interrupts at about 4000 of 10000 steps and asserts `interrupt_at <= resumed_env_steps < budget`.
  So it is not exposed to this race.
- The overshoot (up to ~100% of a 1500-step budget on toy games) is a property of `BatchedCounter`'s 0.5 s flush.
  It does not matter for real budgets.

## Commit
- a4f1ad1 test: resume test counts the resumed budget from the first run's real env steps (FIX-1) (pushed to origin/sp2-game-model)
