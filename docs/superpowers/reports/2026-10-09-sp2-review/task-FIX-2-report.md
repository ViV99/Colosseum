> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/task-FIX-2-report.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# FIX-2 report — flaky `test_sigint_during_startup_exits_130_not_aborted`

## Summary

The -2 came from a CPython 3.12 quirk, not from a missing handler. The repro loop then found two
more ways a Ctrl-C during startup went wrong (one hangs the test, one prints a traceback with
exit 1). All three are fixed in the product code. The test's own expectation (130,
"Interrupted" / "Received SIGINT", no traceback) is unchanged.

| Mode | Symptom | Cause | Fix |
|---|---|---|---|
| A (the reported flake) | exit **-2**, stderr only `Interrupted` | CPython flags a KeyboardInterrupt that leaves a **string `exec`/`eval`** as "unhandled" even when it is caught later. Under `python -m`, `Py_RunMain` then ends the process with SIGINT instead of our `sys.exit(130)`. `import torch` builds hundreds of dataclass methods with `exec(str)` | `cli._forget_unhandled_keyboard_interrupt()`: a trivial `exec("pass", {})` in the `except KeyboardInterrupt` branch of `_interrupts` resets the flag |
| B | Ctrl-C ignored; training runs on (test: `proc.wait(30)` timeout). stderr `Exception ignored in: <function _WeakValueDictionary...KeyedRef.remove> ... KeyboardInterrupt` | The SIGINT lands in a weakref callback (importlib's module-lock bookkeeping during imports). Python cannot raise there, so it prints the exception as unraisable and drops it | `process.catch_lost_interrupts()` (active inside `_interrupts`) records unraisable KeyboardInterrupts instead of printing them. `ProcessSupervisor.install_signal_handlers` treats a recorded one as a received SIGINT (`train` / distributed roles stop with 130 and "Received SIGINT"). A command that finishes anyway ends with "Interrupted" and 130 |
| C | exit 1 with a traceback: `KeyboardInterrupt` in `install_signal_handlers` → `self._watcher.start()`, then `RuntimeError: cannot join thread before it is started` from `restore_signal_handlers` | The SIGINT lands while the watcher thread starts, **before** the handlers were installed. The KeyboardInterrupt leaves an unstarted thread in `self._watcher`, and the `finally` cleanup in the launcher tries to join it | Install the handlers before starting the watcher (a byte written meanwhile is read once the watcher runs). `self._watcher` is set only after `start()` returns |

## Root cause A — evidence

1. **Reproduced without load**, with the original code. `repro.py` (scratch, see below) starts the
   test's exact command, waits for the marker, sleeps U(0, 6) s and `killpg`s SIGINT:
   `{130 Received SIGINT: 47, 130 Interrupted: 11, -2 Interrupted: 2}`. The two -2 cases came 0.74
   and 0.80 s after the marker, inside the torch import. stderr was just `Interrupted`, so
   `_interrupts` **did** catch the KeyboardInterrupt and call `sys.exit(130)`, and the process
   still died from SIGINT.
2. **The mechanism in isolation** (scratch modules, run with `python -m <mod>`; exit codes read with
   `subprocess`, because bash reports 130 for both a signal death and exit 130):
   - `try: exec("raise KeyboardInterrupt", {}) except KeyboardInterrupt: sys.exit(7)` → **-2**
   - the same with `exec(compile(...))` (a code object, so no `PyRun_String`) → 7
   - a plain `raise KeyboardInterrupt` caught the same way → 7
   - `PyRun_String` called via ctypes → -2 under `-m`; 7 when run as a script file.

   CPython 3.12 explains this. `run_eval_code_obj` (the path for string `exec`/`eval`, i.e.
   `PyRun_String*`) sets `_PyRuntime.signals.unhandled_keyboard_interrupt = 1` when the evaluated
   code ends with a KeyboardInterrupt, even if a caller catches it afterwards. For `-m`,
   `pymain_run_module` takes the SystemExit code but `Py_RunMain` then checks that flag and calls
   `exit_sigint()`. Script files and `-c` go through `PyErr_Print` → `Py_Exit` and skip the check.
   That is why the `colosseum` console script never shows the bug and `python -m colosseum` (the
   tests and the Docker/K8s entrypoints) does. The flag is reset at the **start** of every string
   eval, so a later string `exec` before exit hides the bug. This is part of why it is
   intermittent.
3. **The exec that gets interrupted in the real CLI.** `dbgmain.py` (scratch) wraps
   `builtins.exec`/`eval` and logs any KeyboardInterrupt leaving a string exec. The -2 run logged:
   `torch/export/graph_signature.py:32 @dataclasses.dataclass → dataclasses._create_fn → exec(txt, globals, ns)`
   (source `'def __create_fn__(...'`), then `Interrupted`. Other runs showed the same thing in
   `torch/_library/autograd.py` dataclasses.

## Reproduction method and rates

Scratch dir `<scratch>/fix2/`:
- `repro.py N MAX [MIN]` runs the test's command (`train_cmd(TTT_CONFIG, ..., FOREVER)`, same env and
  marker hook, `start_new_session`). After the marker it sleeps U(MIN, MAX), sends `killpg` SIGINT,
  waits 30 s and classifies (rc, "Interrupted", "Received SIGINT", "Traceback", "Aborted").
- `with_load.sh K cmd` runs K `python -c "while True: pass"` burners (8-core machine) and kills
  them on exit. Checked afterwards: no burner left (`pgrep` exit 1).
- `loop_pytest.sh N` runs the real pytest tests (`-k startup`) N times.
- The original code was run from `git archive HEAD src` (prepended to `PYTHONPATH`, which the
  spawned children inherit).

| Scenario | Original code (HEAD cbc0d4c) | After fix |
|---|---|---|
| No load, delay U(0.3, 1.2) s (import window), N=150 / 200 | **18/150 -2** (12%), **1/150 timeout** (mode B) | **200/200** clean (193 Interrupted, 7 Received SIGINT) |
| Intermediate: fix A only, same window, N=150 | — | 0 -2, but **3/150 timeout** (B) and **1/150 rc 1 + traceback** (C) → B and C fixed too |
| 12 burners, delay U(0, 0.05) s (the test's timing), N=100 / 150 | **3/100 -2** | **150/150** clean |
| 12 burners, real pytest `-k startup` ×40 | — | **40/40** passed |

## Fix

- `src/colosseum/cli.py`
  - `_interrupts`: wraps the command in `catch_lost_interrupts()`, creating the marker after it is
    active. After a normal return, a recorded lost interrupt raises KeyboardInterrupt. The
    `except KeyboardInterrupt` branch calls `_forget_unhandled_keyboard_interrupt()` before
    printing "Interrupted" and calling `sys.exit(130)`.
  - New `_forget_unhandled_keyboard_interrupt()`: `exec("pass", {})`, with a docstring that
    explains the CPython flag.
- `src/colosseum/utils/process.py`
  - New `catch_lost_interrupts()` context manager: a `sys.unraisablehook` wrapper. It records
    KeyboardInterrupt, passes everything else to the previous hook, and restores the previous
    hook on exit.
  - New `take_lost_interrupt()`: returns True once, then clears the record.
  - `ProcessSupervisor.install_signal_handlers`: installs the handlers first, takes over a lost
    interrupt as SIGINT, then starts the watcher and sets `self._watcher` only after `start()`.

`restore_signal_handlers` is unchanged. With `_watcher` None it closes both pipe ends, which is
correct when no thread was started.

## Tests (TDD)

New (6 test items):
- `tests/integration/test_sp2_lifecycle.py::test_ctrl_c_inside_string_exec_during_startup_still_exits_130[exec|eval]`
  (mode A). A `python -m` module adds a command to the real `colosseum.cli.main` group; the command
  raises KeyboardInterrupt inside a string `exec`/`eval`. Expects 130 and stderr exactly
  `Interrupted\n`.
- `tests/integration/test_sp2_lifecycle.py::test_ctrl_c_lost_in_a_weakref_callback_during_startup_still_exits_130`
  (mode B, CLI path). The command raises KeyboardInterrupt inside a weakref callback and then
  finishes. Expects stdout `finished True`, 130, stderr exactly `Interrupted\n`.
- `tests/unit/test_process_lifecycle.py::test_sigint_while_the_watcher_thread_starts_is_recorded_not_raised`
  (mode C). `Thread.start` is patched to `raise_signal(SIGINT)` first. Expects no
  KeyboardInterrupt, `stop_event` set, `received_signal == SIGINT`, and the handler restored.
- `tests/unit/test_process_lifecycle.py::test_interrupt_lost_in_an_unraisable_context_reaches_the_supervisor`
  (mode B, supervisor handoff): no stderr output; the record is consumed.
- `tests/unit/test_process_lifecycle.py::test_catch_lost_interrupts_passes_other_unraisables_on_and_restores_the_hook`.

RED (before the fix):
- `pytest tests/integration/test_sp2_lifecycle.py -k string_exec`: 2 failed,
  `assert -2 == 130`, stderr `'Interrupted\n'`. This is exactly the reported symptom, made
  deterministic.
- `pytest tests/unit/test_process_lifecycle.py -k "watcher_thread_starts or unraisable or lost_interrupts"`:
  3 failed.
  - `Failed: a Ctrl-C during the watcher start escaped as KeyboardInterrupt`, then
    `RuntimeError: cannot join thread before it is started`. This is the production mode-C
    traceback.
  - `AttributeError: ... no attribute 'catch_lost_interrupts'` (×2).
- `pytest ... -k weakref_callback`: 1 failed. stderr `Exception ignored in: <function callback> ... KeyboardInterrupt`,
  `assert 0 == 130`.

GREEN: `pytest tests/integration/test_sp2_lifecycle.py tests/unit/test_process_lifecycle.py -q`:
`33 passed in 42.75s`. `ruff check .`: all checks passed.

## Full suite

`OMP_NUM_THREADS=1 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider` →
`1243 passed, 27 deselected in 341.32s (0:05:41)`, exit 0, zero warnings (no warnings summary;
`grep -ci warning` = 0). `.venv/bin/ruff check .` → All checks passed!
New fast-suite count: **1243** (6 new test items in this task).

## Commit

`b97dfec` fix: Ctrl-C during startup always exits 130 (string-exec flag, lost interrupts, watcher start), pushed to origin/sp2-game-model.

## Files changed

- `src/colosseum/cli.py`
- `src/colosseum/utils/process.py`
- `tests/integration/test_sp2_lifecycle.py`
- `tests/unit/test_process_lifecycle.py`

## Deviations from the brief

- The scope is wider than "the -2": modes B and C are the same test failing on a Ctrl-C during
  startup, and both break the spec (Ctrl+C → 130, no traceback). The repro loop found them after
  fix A, so they are fixed here with their own regression tests. The test itself is unchanged.

## Self-review

- Fix A depends on a CPython implementation detail: every string `exec` resets the flag at its
  start. On an interpreter without the flag, `exec("pass")` is harmless. It runs only after a
  KeyboardInterrupt was caught, so it cannot hide a real unhandled one.
- Mode B, non-`train` commands: a Ctrl-C lost in a callback no longer interrupts the command (it
  never did). The command finishes and then exits 130 with "Interrupted" instead of 0. This is
  honest about the interrupt, but `eval`/`bc` still run to the end. A command that ends with
  SystemExit (e.g. a config error, exit 1) keeps its own code.
- `catch_lost_interrupts` replaces `sys.unraisablehook` for the length of the command (main CLI
  process only; spawned children do not run `_interrupts`). Non-KeyboardInterrupt unraisables
  still reach the previous hook.
- The marker is still created inside `_interrupts`, now after the hook is installed, so the
  test's "SIGINT only after the marker" contract holds.

## Concerns

- Fix A relies on CPython internals (documented in the docstring). If a future CPython changes
  how the flag is reset, the regression test `..._string_exec_..._exits_130` catches it.
- New fast-suite count for the controller's doc update: 1243 (+6).
