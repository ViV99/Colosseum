# SP2 Plan — Part A: Foundations (env contract, observations/actions, config) and Model

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

This part covers spec block 0 (SP1 residuals), block 1 (env contract), block 2 (observations, actions, masks, distributions), block 3 (model protocol and centralized critic) and the config schema of block 9 (`validate_config` is T6.2 in part `03-…`). Tasks: T0.1, T1.1–T1.7, T2.1–T2.4.

- T0.1 fixes the three SP1 residuals in the OLD code.
- T1.1–T1.6 build the env contract: tree utilities, the `Units` space, `ObsSpec`/`ActionSpec` with the mask rules, `GameSpec`/`MultiAgentEnv`/`StepResult` with team outcomes, `EpisodeTracker` (every contract check) and vector envs without auto-reset.
- T1.7 is the config v2 schema.
- T2.1–T2.4 build the model side: decider-aware distributions (`UnitsDist` with `only_if`), the `step`/`unroll` model protocol with `EncoderOutput.aux` and a critic encoder, and `build_model` with role injection and role resolution.

**Read `00-overview.md` first.** Its global constraints, shadow package strategy, file map and interface contract apply to every task below. The contract names are implemented exactly; additions and the few deviations are listed in "Contract notes" at the end of this part.

**Starting point.** Branch `sp2-game-model` with the SP2 spec and the plan committed; SP1 is merged (873 fast tests green). `.venv` exists (`scripts/setup-dev.sh`). Nothing under `src/colosseum/sp2/` exists yet.

**Prototype.** Every file of this part was written and run in a scratch copy of the repository (`/home/viv/.claude/jobs/0db917f9/tmp/plan_partA/repo`, outside the repo; a reference, not a dependency). The plan text is generated from those files. The tasks were then replayed one by one on a fresh copy of the repository: after each task its tests, all earlier tests of this part and ruff were green, and after the last task every touched file was byte-identical to the prototype. With all tasks applied the full fast suite passes (1121 passed) with zero warnings, and ruff is clean.

**Conventions used in this part**

- Commands run from the repository root with `.venv/bin/python` and `.venv/bin/ruff`.
- "In `file`, replace: A with: B" is an exact string replacement. Every A occurs exactly once in the file at the start of that step (after all earlier steps and tasks of this part). "Append to the end of `file`" adds the block verbatim (it starts with the blank lines that separate it from the previous code).
- Shown files are complete: copy them verbatim. Docstrings and comments are part of the code.
- New test files have basenames that are unique across `tests/` (the conftest check enforces it) and import support modules by bare name (`from game_helpers import ...`).
- **Shared test kit `tests/game_helpers.py`.** T1.4 creates it (toy games), T1.5 adds drivers (`sample_legal_action`, `play_episode`), T2.3 adds tiny models (`make_test_model`, `CORE_KINDS`, `RandomPolicy` and their parts). Each addition is shown in full in its task. Later parts extend this module; they must not create near-duplicates under other names (overview, "Cross-part execution notes").
- Every task ends with the full fast suite, ruff and a commit pushed to `origin`.

**Order.** Execute the tasks in the order of this part (it is the overview's index order). T1.7 depends only on T0.1 and may run earlier. T2.3 extends the import block of `tests/game_helpers.py` as T1.5 left it, so it needs T1.4 and T1.5 in addition to T2.2.

---

### Task T0.1: SP1 residuals: second Ctrl+C during the queue release, `bc` with an unreadable data file, queue helpers to `core/ipc.py`

Spec block 0 (CLAUDE.md, "Residuals from the SP1 final review", without the SP5 checkpoint-drainer item). This task changes the OLD code in place; nothing under `colosseum.sp2` exists yet.

1. **Second Ctrl+C.** `Launcher._release_queues` calls `_release_command_queues` after the signal handlers were restored. A `KeyboardInterrupt` raised inside it skipped the `cancel_join_thread()` fallback, so an unread command larger than the pipe buffer (64 KiB) made interpreter exit wait on that queue's feeder thread. Fix: catch `BaseException`, call `cancel_join_thread()` on every queue whose feeder thread was not seen to exit, re-raise.
2. **`colosseum bc` with an unreadable file.** `OfflineBCTrainer.load_data` turns `OSError` (`PermissionError`, `IsADirectoryError`, ...) from `torch.load` into a `DataError`, so the CLI prints one `Config error:` line instead of a traceback.
3. **Queue helpers move to `core/ipc.py`.** `_QueueReader`, `_release_command_queues` and `_queue_depths` become the public `QueueReader`, `release_command_queues` and `queue_depths` of `colosseum.core.ipc`, so the SP2 launcher copy (T5.4) imports them instead of copying them. The `registry._reset_mask_row` duplication is not touched: it disappears with the old registry in T7.3.

**Files:**
- Modify: `src/colosseum/core/ipc.py` (import block, `logger`, new section at the end)
- Modify: `src/colosseum/launcher.py` (imports; delete `_QueueReader`, `_release_command_queues`, `_queue_depths`; rename their uses)
- Modify: `src/colosseum/bc/offline_bc.py` (`OfflineBCTrainer.load_data`)
- Create: `tests/unit/test_ipc_queue_helpers.py`
- Modify: `tests/integration/test_bc_cli.py`

**Interfaces:**
- Consumes: SP1 code as merged (`launcher.py`, `core/ipc.py`, `bc/offline_bc.py`, `colosseum.core.errors.DataError`).
- Produces (all in `colosseum.core.ipc`, reused unchanged by `colosseum.sp2`):
  - `class QueueReader(q, label: str, dead_producers: Callable[[], list[str]])` with `busy` (property), `abandoned` (attribute), `drain(wait: float = 0.05) -> list`, `abandon() -> None`, `release(timeout: float) -> bool` (SP1 `_QueueReader` behavior);
  - `release_command_queues(queues: list, timeout: float) -> list` (returns the detached queues; on any `BaseException` detaches every queue whose feeder was not joined and re-raises);
  - `queue_depths(queues: dict[str, Any]) -> dict[str, int]`.

- [ ] **Step 1: Write the failing tests**

The unit tests use fake queues to raise `KeyboardInterrupt` at two points of the release (while draining a queue, and while waiting for a feeder thread).

Create `tests/unit/test_ipc_queue_helpers.py`:

```python
"""Queue helpers live in core/ipc.py; a second Ctrl+C during the release detaches feeders (SP2 T0.1)."""
import multiprocessing as mp
import queue

import numpy as np
import pytest

import colosseum.launcher as launcher_module
from colosseum.core import ipc
from colosseum.core.ipc import QueueReader, queue_depths, release_command_queues


class _FakeCommandQueue:
    """Records the release calls; ``interrupt_in`` names the method that raises KeyboardInterrupt."""

    def __init__(self, name: str, interrupt_in: str | None = None) -> None:
        self.name = name
        self.interrupt_in = interrupt_in
        self.calls: list[str] = []

    def _call(self, what: str) -> None:
        self.calls.append(what)
        if self.interrupt_in == what:
            raise KeyboardInterrupt

    def full(self) -> bool:
        self._call("full")
        return False

    def get(self, timeout=None):
        raise queue.Empty

    def close(self) -> None:
        self._call("close")

    def join_thread(self) -> None:
        self._call("join_thread")

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel_join_thread")


class _SyncJoiner:
    """Stand-in for threading.Thread: runs the target on start(); join() of queue ``b`` is interrupted."""

    def __init__(self, target, name=None, daemon=None) -> None:
        self._target = target

    def start(self) -> None:
        self._target()

    def join(self, timeout=None) -> None:
        if self._target.__self__.name == "b":
            raise KeyboardInterrupt

    def is_alive(self) -> bool:
        return False


def test_helpers_moved_out_of_the_launcher():
    for name in ("_QueueReader", "_release_command_queues", "_queue_depths"):
        assert not hasattr(launcher_module, name)
    assert launcher_module.QueueReader is QueueReader
    assert launcher_module.release_command_queues is release_command_queues


def test_interrupt_while_draining_detaches_every_queue_and_reraises():
    queues = [_FakeCommandQueue("a"), _FakeCommandQueue("b", interrupt_in="full"), _FakeCommandQueue("c")]
    with pytest.raises(KeyboardInterrupt):
        release_command_queues(queues, timeout=0.5)
    # a was closed but its feeder was never seen to exit; b and c were not reached: all are detached.
    assert queues[0].calls[:3] == ["full", "full", "close"] and "cancel_join_thread" in queues[0].calls
    assert queues[1].calls == ["full", "cancel_join_thread"]
    assert queues[2].calls == ["cancel_join_thread"]


def test_interrupt_while_joining_keeps_joined_queues_and_detaches_the_rest(monkeypatch):
    monkeypatch.setattr(ipc.threading, "Thread", _SyncJoiner)
    queues = [_FakeCommandQueue("a"), _FakeCommandQueue("b"), _FakeCommandQueue("c")]
    with pytest.raises(KeyboardInterrupt):
        release_command_queues(queues, timeout=0.5)
    drained = ["full", "full", "close", "join_thread"]
    assert queues[0].calls == drained                                 # joined: left alone
    assert queues[1].calls == [*drained, "cancel_join_thread"]
    assert queues[2].calls == [*drained, "cancel_join_thread"]


def test_release_reads_back_an_unread_large_command_and_joins_the_feeder():
    q = mp.Queue(maxsize=1)
    q.put({"payload": np.zeros(200_000, dtype=np.uint8)})  # > 64 KiB: the feeder blocks in the pipe write
    assert release_command_queues([q], timeout=2.0) == []
    assert q._closed
    assert q._thread is None or not q._thread.is_alive()


def test_queue_reader_drains_a_plain_queue_synchronously():
    q: queue.Queue = queue.Queue()
    for i in range(3):
        q.put(i)
    reader = QueueReader(q, "results", dead_producers=lambda: [])
    assert reader.drain() == [0, 1, 2]
    assert reader.drain() == []
    assert not reader.busy


def test_queue_depths_reports_sizes_and_unknown():
    class _NoSize:
        def qsize(self):
            raise NotImplementedError

    q: queue.Queue = queue.Queue()
    q.put(1)
    assert queue_depths({"a": q, "b": _NoSize()}) == {"a": 1, "b": -1}
```

In `tests/integration/test_bc_cli.py`, replace:

```python
def _empty_dir(tmp_path):
```

with:

```python
def _subdir_named_like_data(tmp_path):
    path = tmp_path / "data_dir"
    (path / "sub.pt").mkdir(parents=True)  # torch.load on a directory: IsADirectoryError (an OSError)
    return path


def _empty_dir(tmp_path):
```

In `tests/integration/test_bc_cli.py`, replace:

```python
    (_empty_dir, "no .pt files"),
], ids=["unreadable", "missing-key", "action-check", "empty-dir"])
```

with:

```python
    (_empty_dir, "no .pt files"),
    (_subdir_named_like_data, "cannot read the BC data file (IsADirectoryError"),
], ids=["unreadable", "missing-key", "action-check", "empty-dir", "os-error"])
```

Append to the end of `tests/integration/test_bc_cli.py`:

```python


def test_bc_cli_reports_a_permission_error_as_one_line_config_error(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CONFIG)
    data = tmp_path / "data.pt"
    _write_data(data)

    def _denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(data))

    monkeypatch.setattr(torch, "load", _denied)
    out = tmp_path / "bc.pt"
    result = CliRunner().invoke(main, ["bc", "-c", str(cfg), "-d", str(data), "-o", str(out)])
    assert result.exit_code == 1, result.output
    assert result.stderr.startswith("Config error:"), result.stderr
    assert "cannot read the BC data file (PermissionError: Permission denied)" in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Traceback" not in result.output and isinstance(result.exception, SystemExit)
    assert not out.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_ipc_queue_helpers.py tests/integration/test_bc_cli.py -q`
Expected: collection error `ImportError: cannot import name 'QueueReader' from 'colosseum.core.ipc'`; with `tests/integration/test_bc_cli.py` alone, `os-error` and `test_bc_cli_reports_a_permission_error_as_one_line_config_error` fail (`result.exception` is an `IsADirectoryError` / `PermissionError`, not `SystemExit`).

- [ ] **Step 3: Move the queue helpers into `core/ipc.py` and guard the release**

In `src/colosseum/core/ipc.py`, replace:

```python
  non-writable-array warning.
"""
```

with:

```python
  non-writable-array warning.

The module also holds the main process's queue helpers (``QueueReader``,
``release_command_queues``, ``queue_depths``), moved here from ``launcher.py``.
"""
```

In `src/colosseum/core/ipc.py`, replace:

```python
import dataclasses
import multiprocessing as mp
import queue
import time
from typing import Any

import numpy as np
import torch
```

with:

```python
import dataclasses
import logging
import multiprocessing as mp
import multiprocessing.queues
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
```

Append to the end of `src/colosseum/core/ipc.py` (the bodies of `QueueReader` and `queue_depths` are SP1's, unchanged; `release_command_queues` wraps SP1's body, now `_release_command_queues`, which also records the joined queues):

```python


# ---------------------------------------------------------------------------
# Queue helpers of the main process (moved from launcher.py in SP2 T0.1)
# ---------------------------------------------------------------------------


class QueueReader:
    """Drains one ``mp.Queue`` on a helper thread, so the main process never blocks on it.

    ``mp.Queue.get`` reads a whole message in ``recv_bytes`` and ignores any timeout once
    the first bytes are there. A producer killed mid-message (OOM, SIGKILL) therefore
    blocks the reader forever: the main process holds the pipe's write end too, so no EOF
    ever arrives. Here ``drain`` waits at most ``wait`` seconds; a read still in progress
    (a large payload arriving or being unpickled) is collected by a later call. Only the
    shutdown gives up on it (``abandon``, after its deadline), once every producer is gone.

    After ``abandon`` no further item can arrive: the dead producer was killed inside the
    feeder's ``send_bytes``, i.e. while holding the queue's cross-process write lock, so no
    other producer can ever write to that pipe again. Should the stuck read finish anyway,
    the items it got are logged as dropped.
    """

    def __init__(self, q: Any, label: str, dead_producers: Callable[[], list[str]]) -> None:
        self._q = q
        self._label = label
        self._dead_producers = dead_producers
        self._thread: threading.Thread | None = None
        self._items: list = []
        self._error: BaseException | None = None
        self._salvaged: list = []
        self.abandoned = False

    @property
    def busy(self) -> bool:
        """A read is in progress (possibly stuck on an incomplete message)."""
        return self._thread is not None and self._thread.is_alive()

    def _read(self) -> None:
        try:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError) as e:
                    if self.abandoned:  # ended by release(): the incomplete item was already reported
                        logger.debug(f"Abandoned read on the {self._label} queue ended: {e!r}")
                    else:
                        logger.warning(f"Dropped an unreadable {self._label} queue item: {e!r}")
                    break
                self._items.append(item)
        except BaseException as e:  # noqa: BLE001 - re-raised by drain() in the main thread
            self._error = e
        if self.abandoned and self._items:
            logger.warning(f"{len(self._items)} item(s) completed on the abandoned {self._label} queue "
                           f"after shutdown gave up on it; dropped")

    def _take(self) -> list:
        items, self._items = self._items, []
        return items

    def drain(self, wait: float = 0.05) -> list:
        """Items read so far; starts a read when the queue has data.

        A new read gets up to ``wait`` seconds to finish, so small items come back from this
        call; a read already in progress is only checked, never waited for, so a stuck
        reader costs nothing per poll. Non-``mp.Queue`` objects (e.g. ``queue.Queue``),
        whose ``get_nowait`` never blocks, are read synchronously.
        """
        if self.abandoned:
            salvaged, self._salvaged = self._salvaged, []
            return salvaged  # complete items read before the stuck message
        if not isinstance(self._q, mp.queues.Queue):
            self._read()
        else:
            if self._thread is None:
                if self._q.empty():
                    return []
                self._thread = threading.Thread(target=self._read, name=f"drain-{self._label}", daemon=True)
                self._thread.start()
                self._thread.join(wait)
            if self._thread.is_alive():
                return []  # still reading: a large item is arriving, or the message is incomplete
            self._thread = None
        error, self._error = self._error, None
        items = self._take()
        if error is not None:
            raise error
        return items

    def abandon(self) -> None:
        """Give up on a read that never completed; the next ``drain`` returns the items read before it."""
        self._salvaged = self._take()
        self.abandoned = True
        dead = self._dead_producers()
        source = f" ({', '.join(dead)} died while sending)" if dead else ""
        logger.error(f"An incomplete item on the {self._label} queue was never completed{source}; "
                     f"it and the rest of that queue are skipped")

    def release(self, timeout: float) -> bool:
        """End an abandoned read; True once no read thread is left (``timeout`` bounds the wait).

        An abandoned read means every producer is gone, so this process holds the pipe's last
        write end (it never writes to the queues it reads): closing it makes the stuck
        ``recv_bytes`` hit end-of-file and the thread exit. The queue can then be closed and
        freed on the calling thread, rather than its semaphores being finalized with this
        daemon thread at interpreter exit. Should a producer still hold a write end, the
        read stays stuck and False is returned.
        """
        if self.busy and self.abandoned:
            try:
                self._q._writer.close()
            except (AttributeError, OSError):
                pass
            self._thread.join(timeout)
        return not self.busy


def _cancel_join(q: Any) -> None:
    try:
        q.cancel_join_thread()
    except (AttributeError, OSError):
        pass


def release_command_queues(queues: list, timeout: float) -> list:
    """Close the queues this process writes to (worker command queues, maxsize 1) and wait,
    bounded by ``timeout`` per step, until their feeder threads have exited.

    A feeder thread still running when its queue is freed, or at interpreter exit, drops the
    last references to the queue's semaphores on that daemon thread, which then unlinks them
    during interpreter shutdown ("leaked semaphore objects" warning). A command its worker
    never took would keep the feeder busy (a large one blocks in the pipe write), so it is
    read back first. A queue still full at the deadline (its reader died holding the read
    lock) or whose feeder does not exit in time is detached instead (``cancel_join_thread``,
    the previous behaviour). Returns the queues that were detached.

    An exception raised meanwhile, including a ``KeyboardInterrupt`` from a second Ctrl+C
    (the launcher has already restored the default signal handlers here), detaches every
    queue whose feeder was not seen to exit and is then re-raised: otherwise interpreter
    exit would wait on the feeder of an unread command larger than the pipe buffer.
    """
    finished: set[int] = set()  # id(q) of queues whose feeder thread was joined
    try:
        return _release_command_queues(queues, timeout, finished)
    except BaseException:
        for q in queues:
            if id(q) not in finished:
                _cancel_join(q)
        raise


def _release_command_queues(queues: list, timeout: float, finished: set[int]) -> list:
    deadline = time.monotonic() + timeout
    detached: list = []
    joiners: list[tuple[Any, threading.Thread]] = []
    for q in queues:
        try:
            while q.full() and time.monotonic() < deadline:
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            if q.full():
                detached.append(q)
                continue
            q.close()
        except Exception as e:  # noqa: BLE001 - best effort: a release never replaces the run's outcome
            logger.debug(f"Could not drain and close a command queue: {e!r}")
            detached.append(q)
            continue
        joiner = threading.Thread(target=q.join_thread, name="queue-release", daemon=True)
        joiner.start()
        joiners.append((q, joiner))
    join_deadline = time.monotonic() + timeout
    for q, joiner in joiners:
        joiner.join(max(0.0, join_deadline - time.monotonic()))
        if joiner.is_alive():
            detached.append(q)
        else:
            finished.add(id(q))
    for q in detached:
        _cancel_join(q)
    if detached:
        logger.debug(f"{len(detached)} command queue(s) could not be drained and closed in time; "
                     f"their feeder threads are detached")
    return detached


def queue_depths(queues: dict[str, Any]) -> dict[str, int]:
    """Approximate items waiting per queue (-1 where the platform cannot tell)."""
    depths = {}
    for name, q in queues.items():
        try:
            depths[name] = int(q.qsize())
        except (NotImplementedError, OSError):
            depths[name] = -1
    return depths
```

- [ ] **Step 4: Use the moved helpers in the launcher**

In `src/colosseum/launcher.py`, replace:

```python
import logging
import multiprocessing as mp
import multiprocessing.queues
import queue
import signal
import threading
import time
```

with:

```python
import logging
import multiprocessing as mp
import queue
import signal
import time
```

In `src/colosseum/launcher.py`, replace:

```python
from colosseum.core.ipc import SharedCounter
```

with:

```python
from colosseum.core.ipc import QueueReader, SharedCounter, queue_depths, release_command_queues
```

In `src/colosseum/launcher.py`, delete everything from the line `class _QueueReader:` up to, but not including, these lines (this removes `_QueueReader`, `_release_command_queues` and `_queue_depths`; the banner stays):

```python
# =====================================================================
# Launcher
```

In `src/colosseum/launcher.py`, rename (all occurrences):

- every `_QueueReader` -> `QueueReader`
- every `_release_command_queues` -> `release_command_queues`
- every `_queue_depths(` -> `queue_depths(`

`threading` and `multiprocessing.queues` are no longer used in `launcher.py` (ruff F401 would flag them); `queue` still is (`queue.Full` in `_refresh_worker_matches`).

- [ ] **Step 5: Report unreadable BC data files as a `DataError`**

In `src/colosseum/bc/offline_bc.py`, replace:

```python
            except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
                detail = (str(e).strip().splitlines() or [""])[0]
```

with:

```python
            except OSError as e:  # PermissionError, IsADirectoryError, FileNotFoundError, I/O errors
                reason = e.strerror or str(e) or type(e).__name__
                raise DataError(f"{f}: cannot read the BC data file ({type(e).__name__}: {reason})") from e
            except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
                detail = (str(e).strip().splitlines() or [""])[0]
```

The CLI already maps `DataError` (a `ConfigError`) to one `Config error:` line with exit code 1 (`_config_errors` in `cli.py`).

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_ipc_queue_helpers.py tests/integration/test_bc_cli.py -q`
Expected: `15 passed`.

- [ ] **Step 7: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/core/ipc.py \
  src/colosseum/launcher.py \
  src/colosseum/bc/offline_bc.py \
  tests/unit/test_ipc_queue_helpers.py \
  tests/integration/test_bc_cli.py
git commit -m "fix: SP1 residuals: detach command queues on a second Ctrl+C, one-line bc data errors, queue helpers in core/ipc"
git push origin sp2-game-model
```

---

### Task T1.1: Tree utilities

Spec block 2, "Деревья". Observations, actions, masks and global states become trees (`np.ndarray | Tensor | dict[str, Tree]`, dict insertion order significant). One module holds the generic utilities; model states keep using `colosseum.networks.state` (unchanged). This task also creates the `colosseum.sp2` shadow package (overview, "Shadow package strategy").

**Files:**
- Create: `src/colosseum/sp2/__init__.py`, `src/colosseum/sp2/core/__init__.py` (empty)
- Create: `src/colosseum/sp2/core/tree.py`
- Test: `tests/unit/test_tree_utils.py`

**Interfaces:**
- Consumes: `colosseum.core.ipc.numpy_to_tensor`, `colosseum.core.ipc.tensor_to_numpy` (SP1).
- Produces (`colosseum.sp2.core.tree`): `Tree`, `tree_map(fn, tree, *rest)`, `tree_leaves(tree)`, `tree_paths(tree)`, `tree_get(tree, path)`, `tree_stack(trees, axis=0)`, `tree_index(tree, idx)`, `tree_assign(dst, idx, src)`, `tree_to_torch(tree, device="cpu")`, `tree_to_numpy(tree)`, `tree_same_structure(a, b)` — exactly the contract signatures. Details later tasks rely on:
  - `tree_map` raises `ValueError` naming the path when dict keys differ; key order of `rest` trees does not matter, the result follows `tree`;
  - `None` is a leaf; `tree_to_torch(None)` / `tree_to_numpy(None)` return `None`;
  - `tree_assign` needs a dict `dst` (a bare array is assigned directly by the caller), `TypeError` otherwise;
  - `tree_get(tree, ())` returns `tree`; a missing path raises `KeyError` naming it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_tree_utils.py`:

```python
"""Tree utilities for observation / action / mask trees (SP2 T1.1)."""
import numpy as np
import pytest
import torch

from colosseum.sp2.core.tree import (
    tree_assign,
    tree_get,
    tree_index,
    tree_leaves,
    tree_map,
    tree_paths,
    tree_same_structure,
    tree_stack,
    tree_to_numpy,
    tree_to_torch,
)


def _obs():
    # Insertion order (grid before vec, b before a) is deliberately not alphabetical.
    return {
        "grid": np.arange(4, dtype=np.uint8).reshape(2, 2),
        "vec": {"b": np.array([1.0, 2.0], dtype=np.float32), "a": np.array(3, dtype=np.int64)},
    }


def test_leaves_and_paths_follow_insertion_order():
    obs = _obs()
    assert tree_paths(obs) == [("grid",), ("vec", "b"), ("vec", "a")]
    leaves = tree_leaves(obs)
    assert [leaf.dtype for leaf in leaves] == [np.uint8, np.float32, np.int64]
    assert tree_paths(np.zeros(3)) == [()]
    assert len(tree_leaves(np.zeros(3))) == 1


def test_tree_map_aligns_leaves_and_keeps_the_first_trees_order():
    a = {"x": np.array([1, 2]), "y": np.array([3])}
    b = {"y": np.array([10]), "x": np.array([20, 30])}
    out = tree_map(lambda u, v: u + v, a, b)
    assert list(out) == ["x", "y"]
    assert out["x"].tolist() == [21, 32] and out["y"].tolist() == [13]


def test_tree_map_rejects_different_structures_with_the_path():
    with pytest.raises(ValueError, match="vec"):
        tree_map(lambda u, v: u, _obs(), {"grid": np.zeros(1), "vec": np.zeros(1)})
    with pytest.raises(ValueError, match="keys"):
        tree_map(lambda u, v: u, {"a": 1}, {"b": 1})


def test_tree_get():
    obs = _obs()
    assert tree_get(obs, ("vec", "a")) == 3
    assert tree_get(obs, ()) is obs
    with pytest.raises(KeyError, match="vec/c"):
        tree_get(obs, ("vec", "c"))


def test_tree_stack_numpy_preserves_dtypes_and_torch_stacks_tensors():
    stacked = tree_stack([_obs(), _obs(), _obs()])
    assert stacked["grid"].shape == (3, 2, 2) and stacked["grid"].dtype == np.uint8
    assert stacked["vec"]["a"].shape == (3,) and stacked["vec"]["a"].dtype == np.int64
    t = tree_stack([{"x": torch.zeros(2)}, {"x": torch.ones(2)}], axis=1)
    assert isinstance(t["x"], torch.Tensor) and t["x"].shape == (2, 2)
    assert t["x"][:, 1].tolist() == [1.0, 1.0]
    with pytest.raises(ValueError):
        tree_stack([])


def test_tree_index_and_assign():
    buf = {"grid": np.zeros((5, 2, 2), dtype=np.uint8),
           "vec": {"b": np.zeros((5, 2), dtype=np.float32), "a": np.zeros(5, dtype=np.int64)}}
    tree_assign(buf, 3, _obs())
    row = tree_index(buf, 3)
    assert row["grid"].tolist() == [[0, 1], [2, 3]] and row["grid"].dtype == np.uint8
    assert row["vec"]["b"].tolist() == [1.0, 2.0] and int(row["vec"]["a"]) == 3
    assert buf["grid"][2].sum() == 0
    with pytest.raises(TypeError):
        tree_assign(np.zeros(3), 0, np.ones(()))


def test_torch_numpy_round_trip_preserves_dtypes():
    t = tree_to_torch(_obs())
    assert t["grid"].dtype == torch.uint8 and t["vec"]["a"].dtype == torch.int64
    back = tree_to_numpy(t)
    assert back["grid"].dtype == np.uint8 and back["vec"]["b"].dtype == np.float32
    assert np.array_equal(back["grid"], _obs()["grid"])
    assert tree_to_torch(None) is None and tree_to_numpy(None) is None


def test_tree_same_structure_is_order_sensitive():
    assert tree_same_structure(_obs(), _obs())
    assert not tree_same_structure({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not tree_same_structure({"a": 1}, np.zeros(1))
    assert tree_same_structure(np.zeros(1), torch.zeros(2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_tree_utils.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2'`.

- [ ] **Step 3: Create the shadow package and the tree module**

Create the empty package files:

```bash
touch src/colosseum/sp2/__init__.py
touch src/colosseum/sp2/core/__init__.py
```

Create `src/colosseum/sp2/core/tree.py`:

```python
"""Tree utilities for observations, actions, masks and global state (SP2 spec block 2).

A ``Tree`` is a leaf (``np.ndarray``, ``torch.Tensor``, a number or ``None``) or a
``dict[str, Tree]``. Dict insertion order is significant: it is the natural order of
the gymnasium space the tree comes from (``Dict.spaces`` order), and every function
here walks dicts in that order. Model states keep their own helpers in
:mod:`colosseum.networks.state`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from colosseum.core.ipc import numpy_to_tensor, tensor_to_numpy

Tree = Any  # np.ndarray | torch.Tensor | number | None | dict[str, Tree]


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _map(fn: Callable[..., Any], path: tuple[str, ...], tree: Tree, rest: tuple[Tree, ...]) -> Tree:
    if isinstance(tree, dict):
        for other in rest:
            if not isinstance(other, dict) or other.keys() != tree.keys():
                got = sorted(other) if isinstance(other, dict) else type(other).__name__
                raise ValueError(f"tree structures differ at {_fmt(path)}: keys {list(tree)} vs {got}")
        return {k: _map(fn, (*path, k), tree[k], tuple(o[k] for o in rest)) for k in tree}
    for other in rest:
        if isinstance(other, dict):
            raise ValueError(f"tree structures differ at {_fmt(path)}: a leaf vs a dict with keys {list(other)}")
    return fn(tree, *rest)


def tree_map(fn: Callable[..., Any], tree: Tree, *rest: Tree) -> Tree:
    """Apply ``fn(leaf, *rest_leaves)`` to aligned leaves; the result has the structure of ``tree``.

    ``rest`` trees must have the same dict keys at every node (``ValueError`` naming the
    path otherwise); their key order does not matter.
    """
    return _map(fn, (), tree, rest)


def tree_leaves(tree: Tree) -> list[Any]:
    """All leaves, depth-first in dict insertion order."""
    out: list[Any] = []
    tree_map(lambda leaf: out.append(leaf), tree)
    return out


def tree_paths(tree: Tree) -> list[tuple[str, ...]]:
    """Path of every leaf, in :func:`tree_leaves` order; ``()`` for a bare leaf."""
    out: list[tuple[str, ...]] = []

    def walk(node: Tree, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, (*path, k))
        else:
            out.append(path)

    walk(tree, ())
    return out


def tree_get(tree: Tree, path: tuple[str, ...]) -> Any:
    """The subtree at ``path`` (``KeyError`` naming the path if it does not exist)."""
    node = tree
    for i, key in enumerate(path):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"no node {_fmt(path[: i + 1])} in the tree")
        node = node[key]
    return node


def _stack_leaves(leaves: Sequence[Any], axis: int) -> Any:
    if all(isinstance(x, torch.Tensor) for x in leaves):
        return torch.stack(list(leaves), dim=axis)
    if any(isinstance(x, torch.Tensor) for x in leaves):
        raise TypeError("tree_stack: cannot stack torch tensors with numpy arrays")
    return np.stack([np.asarray(x) for x in leaves], axis=axis)


def tree_stack(trees: Sequence[Tree], axis: int = 0) -> Tree:
    """Stack aligned leaves of ``trees`` along a new ``axis`` (``torch.stack`` for tensors,
    ``np.stack`` otherwise; numpy dtypes are preserved)."""
    trees = list(trees)
    if not trees:
        raise ValueError("tree_stack needs at least one tree")
    return tree_map(lambda *leaves: _stack_leaves(leaves, axis), trees[0], *trees[1:])


def tree_index(tree: Tree, idx: Any) -> Tree:
    """``leaf[idx]`` for every leaf."""
    return tree_map(lambda leaf: leaf[idx], tree)


def tree_assign(dst: Tree, idx: Any, src: Tree) -> None:
    """In place: ``dst_leaf[idx] = src_leaf`` for every aligned leaf (numpy casts to dst's dtype)."""
    if not isinstance(dst, dict):
        raise TypeError("tree_assign: dst must be a dict of arrays (a bare array cannot be assigned "
                        "through a function argument); index it directly")

    def assign(d: Any, s: Any) -> None:
        d[idx] = s

    tree_map(assign, dst, src)


def tree_to_torch(tree: Tree, device: str | torch.device = "cpu") -> Tree:
    """numpy leaves -> tensors on ``device`` (zero copy on CPU when possible); dtypes preserved.

    Tensor leaves are moved to ``device``; ``None`` leaves stay ``None``.
    """
    def convert(leaf: Any) -> Any:
        if leaf is None:
            return None
        if isinstance(leaf, torch.Tensor):
            return leaf.to(device)
        return numpy_to_tensor(np.asarray(leaf)).to(device)

    return tree_map(convert, tree)


def tree_to_numpy(tree: Tree) -> Tree:
    """Tensor leaves -> detached CPU numpy copies; numpy leaves are kept; ``None`` stays ``None``."""
    def convert(leaf: Any) -> Any:
        if leaf is None:
            return None
        if isinstance(leaf, torch.Tensor):
            return tensor_to_numpy(leaf)
        return np.asarray(leaf)

    return tree_map(convert, tree)


def tree_same_structure(a: Tree, b: Tree) -> bool:
    """True if ``a`` and ``b`` have the same dict keys in the same order at every node."""
    if isinstance(a, dict) != isinstance(b, dict):
        return False
    if not isinstance(a, dict):
        return True
    if list(a) != list(b):
        return False
    return all(tree_same_structure(a[k], b[k]) for k in a)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_tree_utils.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/sp2/__init__.py \
  src/colosseum/sp2/core/__init__.py \
  src/colosseum/sp2/core/tree.py \
  tests/unit/test_tree_utils.py
git commit -m "feat(sp2): tree utilities for observation, action and mask trees"
git push origin sp2-game-model
```

---

### Task T1.2: `Units` action space

Spec block 2, "Пространство `Units`". A `gymnasium.spaces.Space` for up to `max_units` units with the same per-unit action: `Discrete`, `MultiDiscrete`, 1-D `Box` or a `Dict` of `Discrete` / 1-D `Box`. Components are named and kept in natural order (the `Dict.spaces` order; a plain `dict` passed to `gymnasium.spaces.Dict` is sorted by gymnasium, a list of pairs keeps its order — a test pins this caveat). `only_if={child: (parent, values)}` gates a component by the chosen value of a discrete parent of the same unit; the constructor rejects unknown names, a non-discrete parent, self-reference, an empty or out-of-range value set and cycles.

**Files:**
- Create: `src/colosseum/sp2/envs/__init__.py` (empty)
- Create: `src/colosseum/sp2/envs/spaces.py`
- Test: `tests/unit/test_units_space.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (`colosseum.sp2.envs.spaces`): `UnitComponent(name, kind, size)`, `Units(max_units, per_unit, only_if=None, seed=None)` with `max_units`, `per_unit`, `per_unit_kind`, `components`, `only_if` (normalized to `dict[str, tuple[str, frozenset[int]]]`), `sample(mask=None)`, `contains(x)` — the contract. Additions: `Units.__eq__` / `__hash__` (value equality: `max_units`, `per_unit`, `only_if`; `GameSpec` and `ActionSpec` equality rely on it) and `__repr__`. `sample(mask)` takes the mask layout of T1.3 (`{"unit": bool[U], "action": bool[U, sum discrete sizes]}`, keys optional) and returns 0 for absent units and empty rows.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_units_space.py`:

```python
"""``Units`` action space: components, natural Dict order, only_if checks, sample/contains (SP2 T1.2)."""
import pickle

import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from colosseum.sp2.envs.spaces import UnitComponent, Units


def _move_target(only_if=None):
    per_unit = Dict([("move", Discrete(4)), ("target", Discrete(3)), ("thrust", Box(-1.0, 1.0, (2,)))])
    return Units(3, per_unit, only_if=only_if, seed=0)


def test_components_for_each_per_unit_kind():
    assert Units(5, Discrete(4)).components == (UnitComponent("0", "discrete", 4),)
    assert Units(5, Discrete(4)).per_unit_kind == "discrete"
    md = Units(5, MultiDiscrete([3, 7]))
    assert md.per_unit_kind == "multi_discrete"
    assert md.components == (UnitComponent("0", "discrete", 3), UnitComponent("1", "discrete", 7))
    box = Units(5, Box(-1.0, 1.0, (2,)))
    assert box.per_unit_kind == "box" and box.components == (UnitComponent("0", "box", 2),)
    d = _move_target()
    assert d.per_unit_kind == "dict"
    assert [c.name for c in d.components] == ["move", "target", "thrust"]
    assert [c.kind for c in d.components] == ["discrete", "discrete", "box"]


def test_dict_component_order_is_the_spaces_order():
    # gymnasium sorts the keys of a plain dict; a list of pairs keeps the given order.
    sorted_keys = Units(2, Dict({"move": Discrete(5), "attack": Discrete(3)}))
    kept = Units(2, Dict([("move", Discrete(5)), ("attack", Discrete(3))]))
    assert [c.name for c in sorted_keys.components] == ["attack", "move"]
    assert [c.name for c in kept.components] == ["move", "attack"]


@pytest.mark.parametrize("per_unit, message", [
    (Dict([("a", MultiDiscrete([2, 2]))]), "only be Discrete or 1-D Box"),
    (Dict([("a", Dict([("b", Discrete(2))]))]), "only be Discrete or 1-D Box"),
    (Box(-1.0, 1.0, (2, 2)), "1-D"),
    (Box(0, 3, (2,), dtype=np.int64), "float dtype"),
    (Discrete(3, start=1), "start"),
    (MultiBinary(3), "must be Discrete, MultiDiscrete, Box or Dict"),
    (Dict([]), "empty Dict"),
])
def test_invalid_per_unit_spaces(per_unit, message):
    with pytest.raises(ValueError, match=message):
        Units(2, per_unit)


def test_max_units_must_be_positive():
    with pytest.raises(ValueError, match="max_units"):
        Units(0, Discrete(2))


@pytest.mark.parametrize("only_if, message", [
    ({"nope": ("move", {3})}, "unknown component 'nope'"),
    ({"target": ("nope", {3})}, "unknown parent 'nope'"),
    ({"target": ("thrust", {0})}, "must be a discrete component"),
    ({"target": ("target", {0})}, "cannot depend on itself"),
    ({"target": ("move", set())}, "empty"),
    ({"target": ("move", {4})}, "outside parent 'move' range"),
    ({"target": ("move", {0}), "move": ("target", {0})}, "cycle"),
    ({"target": "move"}, "must be \\(parent, values\\)"),
])
def test_invalid_only_if(only_if, message):
    with pytest.raises(ValueError, match=message):
        _move_target(only_if)


def test_only_if_is_normalized_to_frozensets():
    space = _move_target({"target": ("move", [3, 3, 1])})
    assert space.only_if == {"target": ("move", frozenset({1, 3}))}
    assert Units(2, MultiDiscrete([3, 4]), only_if={"1": ("0", {2})}).only_if == {"1": ("0", frozenset({2}))}


def test_sample_respects_unit_and_action_masks():
    space = _move_target({"target": ("move", {3})})
    mask = {
        "unit": np.array([True, True, False]),
        # move (4) | target (3); unit 1 has an empty move row, unit 2 is absent
        "action": np.array([[0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1]], dtype=bool),
    }
    for _ in range(20):
        a = space.sample(mask)
        assert a["move"].dtype == np.int64 and a["move"].shape == (3,)
        assert a["move"][0] == 3 and a["target"][0] == 1
        assert a["move"][1] == 0          # empty row -> 0
        assert a["move"][2] == 0 and a["target"][2] == 0 and np.all(a["thrust"][2] == 0.0)
        assert a["thrust"].dtype == np.float32 and a["thrust"].shape == (3, 2)
        assert space.contains(a)


def test_sample_layouts_per_kind():
    assert Units(4, Discrete(3), seed=1).sample().shape == (4,)
    md = Units(4, MultiDiscrete([3, 5]), seed=1).sample()
    assert md.shape == (4, 2) and md.dtype == np.int64 and np.all(md[:, 1] < 5)
    box = Units(4, Box(-2.0, 2.0, (3,)), seed=1).sample()
    assert box.shape == (4, 3) and box.dtype == np.float32 and np.all(np.abs(box) <= 2.0)


def test_contains_checks_structure_dtype_and_range():
    space = Units(2, MultiDiscrete([3, 5]))
    assert space.contains(np.array([[2, 4], [0, 0]]))
    assert not space.contains(np.array([[3, 0], [0, 0]]))            # out of range
    assert not space.contains(np.array([[0, 0]]))                     # wrong U
    assert not space.contains(np.array([[0.0, 0.0], [0.0, 0.0]]))     # float for discrete
    d = _move_target()
    good = {"move": np.zeros(3, np.int64), "target": np.zeros(3, np.int64), "thrust": np.zeros((3, 2), np.float32)}
    assert d.contains(good)
    assert not d.contains({"move": good["move"], "target": good["target"]})            # missing key
    assert not d.contains({**good, "thrust": np.full((3, 2), 5.0, np.float32)})        # outside the Box


def test_equality_and_pickle():
    a = _move_target({"target": ("move", {3})})
    b = _move_target({"target": ("move", [3])})
    assert a == b and a != _move_target() and a != Units(3, Discrete(4))
    assert pickle.loads(pickle.dumps(a)) == a
    assert "Units(3" in repr(a)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_units_space.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.envs'`.

- [ ] **Step 3: Implement `Units`**

Create the empty package files:

```bash
touch src/colosseum/sp2/envs/__init__.py
```

Create `src/colosseum/sp2/envs/spaces.py`:

```python
"""``Units``: an action space of up to ``max_units`` units with the same per-unit action (SP2 block 2).

The action of a ``Units`` group is an array (or a dict of arrays) with a leading unit
dimension ``[U]``:

- ``per_unit = Discrete(A)``: ``int64[U]``;
- ``per_unit = MultiDiscrete([A1..Ac])``: ``int64[U, C]``;
- ``per_unit = Box(d,)``: ``float32[U, d]``;
- ``per_unit = Dict(...)`` of ``Discrete`` / 1-D ``Box``: ``{name: int64[U] | float32[U, d]}``.

The per-unit action is a list of named components in natural order: ``"0"`` for
``Discrete`` / ``Box``, ``"0".."C-1"`` for ``MultiDiscrete``, and ``Dict.spaces`` order for
``Dict``. Note that ``gymnasium.spaces.Dict`` built from a plain ``dict`` sorts its keys;
pass a list of ``(key, space)`` pairs to keep your own order. The order fixes the layout of
the ``action`` mask: ``{"unit": bool[U], "action": bool[U, sum of discrete sizes]}``.

``only_if={child: (parent, values)}``: the child component counts (log-prob, entropy, KL,
loss) only where the chosen value of ``parent`` (a discrete component of the same unit) is
in ``values``. Unit slot indices are assigned by the env; the framework does not link slots
across steps.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium
import numpy as np


@dataclass(frozen=True)
class UnitComponent:
    """One named component of the per-unit action."""

    name: str
    kind: Literal["discrete", "box"]
    size: int  # categories (discrete) or dimension (box)


def _discrete_size(space: gymnasium.spaces.Discrete, what: str) -> int:
    if int(space.start) != 0:
        raise ValueError(f"{what}: Discrete(start={int(space.start)}) is not supported; use start=0")
    return int(space.n)


def _box_size(space: gymnasium.spaces.Box, what: str) -> int:
    if len(space.shape) != 1:
        raise ValueError(f"{what}: Box must be 1-D (shape (d,)), got shape {space.shape}")
    if not np.issubdtype(space.dtype, np.floating):
        raise ValueError(f"{what}: Box must have a float dtype, got {space.dtype}")
    return int(space.shape[0])


class Units(gymnasium.spaces.Space):
    """Per-unit actions for up to ``max_units`` units (see the module docstring)."""

    def __init__(
        self,
        max_units: int,
        per_unit: gymnasium.spaces.Discrete | gymnasium.spaces.MultiDiscrete
        | gymnasium.spaces.Box | gymnasium.spaces.Dict,
        only_if: Mapping[str, tuple[str, Collection[int]]] | None = None,
        seed: int | None = None,
    ) -> None:
        if int(max_units) < 1:
            raise ValueError(f"Units: max_units must be >= 1, got {max_units}")
        self.max_units = int(max_units)
        self.per_unit = per_unit
        components: list[UnitComponent] = []
        self._boxes: dict[str, gymnasium.spaces.Box] = {}
        if isinstance(per_unit, gymnasium.spaces.Discrete):
            self.per_unit_kind: Literal["discrete", "multi_discrete", "box", "dict"] = "discrete"
            components.append(UnitComponent("0", "discrete", _discrete_size(per_unit, "Units per_unit")))
        elif isinstance(per_unit, gymnasium.spaces.MultiDiscrete):
            if per_unit.nvec.ndim != 1:
                raise ValueError(f"Units per_unit: MultiDiscrete must be 1-D, got nvec shape {per_unit.nvec.shape}")
            if np.any(per_unit.start != 0):
                raise ValueError("Units per_unit: MultiDiscrete with a non-zero start is not supported")
            self.per_unit_kind = "multi_discrete"
            components.extend(UnitComponent(str(i), "discrete", int(n)) for i, n in enumerate(per_unit.nvec))
        elif isinstance(per_unit, gymnasium.spaces.Box):
            self.per_unit_kind = "box"
            components.append(UnitComponent("0", "box", _box_size(per_unit, "Units per_unit")))
            self._boxes["0"] = per_unit
        elif isinstance(per_unit, gymnasium.spaces.Dict):
            if not per_unit.spaces:
                raise ValueError("Units per_unit: an empty Dict has no components")
            self.per_unit_kind = "dict"
            for name, sub in per_unit.spaces.items():
                what = f"Units per_unit[{name!r}]"
                if isinstance(sub, gymnasium.spaces.Discrete):
                    components.append(UnitComponent(name, "discrete", _discrete_size(sub, what)))
                elif isinstance(sub, gymnasium.spaces.Box):
                    components.append(UnitComponent(name, "box", _box_size(sub, what)))
                    self._boxes[name] = sub
                else:
                    raise ValueError(f"{what}: Dict values may only be Discrete or 1-D Box, "
                                     f"got {type(sub).__name__}")
        else:
            raise ValueError(f"Units per_unit must be Discrete, MultiDiscrete, Box or Dict, "
                             f"got {type(per_unit).__name__}")
        self.components: tuple[UnitComponent, ...] = tuple(components)
        self.only_if: dict[str, tuple[str, frozenset[int]]] = self._check_only_if(only_if or {})
        super().__init__(shape=None, dtype=None, seed=seed)

    # ------------------------------------------------------------------
    # Construction checks
    # ------------------------------------------------------------------

    def _check_only_if(self, only_if: Mapping[str, tuple[str, Collection[int]]]) -> dict:
        by_name = {c.name: c for c in self.components}
        out: dict[str, tuple[str, frozenset[int]]] = {}
        for child, rule in only_if.items():
            if child not in by_name:
                raise ValueError(f"Units only_if: unknown component {child!r} (components: {list(by_name)})")
            try:
                parent, values = rule
            except (TypeError, ValueError):
                raise ValueError(f"Units only_if[{child!r}] must be (parent, values), got {rule!r}") from None
            if parent not in by_name:
                raise ValueError(f"Units only_if[{child!r}]: unknown parent {parent!r} (components: {list(by_name)})")
            if parent == child:
                raise ValueError(f"Units only_if[{child!r}]: a component cannot depend on itself")
            if by_name[parent].kind != "discrete":
                raise ValueError(f"Units only_if[{child!r}]: parent {parent!r} must be a discrete component")
            allowed = frozenset(int(v) for v in values)
            if not allowed:
                raise ValueError(f"Units only_if[{child!r}]: the set of parent values is empty")
            n = by_name[parent].size
            bad = sorted(v for v in allowed if not 0 <= v < n)
            if bad:
                raise ValueError(f"Units only_if[{child!r}]: values {bad} are outside parent {parent!r} range [0, {n})")
            out[child] = (parent, allowed)
        for start in out:  # every chain child -> parent -> ... must end: no cycles
            seen = {start}
            node = start
            while node in out:
                node = out[node][0]
                if node in seen:
                    raise ValueError(f"Units only_if: cycle through component {node!r}")
                seen.add(node)
        return out

    # ------------------------------------------------------------------
    # gymnasium.Space API
    # ------------------------------------------------------------------

    @property
    def is_np_flattenable(self) -> bool:
        return False

    def _component_values(self, x: Any) -> dict[str, np.ndarray] | None:
        """``x`` split into component arrays ``[U]`` / ``[U, d]``, or None if the structure is wrong."""
        U = self.max_units
        if self.per_unit_kind == "dict":
            if not isinstance(x, Mapping) or set(x) != {c.name for c in self.components}:
                return None
            return {c.name: np.asarray(x[c.name]) for c in self.components}
        arr = np.asarray(x)
        if self.per_unit_kind == "discrete":
            return {"0": arr} if arr.shape == (U,) else None
        if self.per_unit_kind == "multi_discrete":
            if arr.shape != (U, len(self.components)):
                return None
            return {c.name: arr[:, i] for i, c in enumerate(self.components)}
        return {"0": arr}  # box

    def contains(self, x: Any) -> bool:
        values = self._component_values(x)
        if values is None:
            return False
        U = self.max_units
        for c in self.components:
            v = values[c.name]
            if c.kind == "discrete":
                if v.shape != (U,) or not np.issubdtype(v.dtype, np.integer):
                    return False
                if np.any(v < 0) or np.any(v >= c.size):
                    return False
            else:
                box = self._boxes[c.name]
                if v.shape != (U, c.size) or not np.issubdtype(v.dtype, np.floating):
                    return False
                if np.any(v < box.low) or np.any(v > box.high):
                    return False
        return True

    def sample(self, mask: Any = None, probability: Any = None) -> Any:
        """A random action. ``mask`` (``{"unit", "action"}``, keys optional) restricts the discrete
        components to legal values; units with ``unit=False`` and empty rows give 0."""
        if probability is not None:
            raise NotImplementedError("Units.sample does not support `probability`")
        U = self.max_units
        rng = self.np_random
        unit = np.ones(U, dtype=bool)
        action = None
        if mask is not None:
            if mask.get("unit") is not None:
                unit = np.asarray(mask["unit"], dtype=bool)
            if mask.get("action") is not None:
                action = np.asarray(mask["action"], dtype=bool)
        values: dict[str, np.ndarray] = {}
        offset = 0
        for c in self.components:
            if c.kind == "discrete":
                out = np.zeros(U, dtype=np.int64)
                for u in range(U):
                    if not unit[u]:
                        continue
                    legal = np.arange(c.size) if action is None else np.flatnonzero(action[u, offset:offset + c.size])
                    if legal.size:
                        out[u] = int(rng.choice(legal))
                values[c.name] = out
                offset += c.size
            else:
                box = self._boxes[c.name]
                low = np.broadcast_to(box.low, (c.size,)).astype(np.float64)
                high = np.broadcast_to(box.high, (c.size,)).astype(np.float64)
                bounded = np.isfinite(low) & np.isfinite(high)
                draw = np.where(bounded, rng.uniform(np.where(bounded, low, 0.0), np.where(bounded, high, 1.0),
                                                     size=(U, c.size)),
                                rng.normal(size=(U, c.size)))
                draw = np.clip(draw, low, high).astype(np.float32)
                draw[~unit] = 0.0
                values[c.name] = draw
        if self.per_unit_kind == "dict":
            return values
        if self.per_unit_kind == "multi_discrete":
            return np.stack([values[c.name] for c in self.components], axis=1)
        return values["0"]

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Units) and self.max_units == other.max_units
                and self.per_unit == other.per_unit and self.only_if == other.only_if)

    def __hash__(self) -> int:
        return hash((self.max_units, repr(self.per_unit)))

    def __repr__(self) -> str:
        rule = f", only_if={ {k: (p, sorted(v)) for k, (p, v) in self.only_if.items()} }" if self.only_if else ""
        return f"Units({self.max_units}, {self.per_unit!r}{rule})"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_units_space.py -q`
Expected: `23 passed`.

- [ ] **Step 5: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/sp2/envs/__init__.py \
  src/colosseum/sp2/envs/spaces.py \
  tests/unit/test_units_space.py
git commit -m "feat(sp2): Units action space with per-unit components and only_if"
git push origin sp2-game-model
```

---

### Task T1.3: `ObsSpec`, `ActionSpec` and the mask rules

Spec block 2, "Наблюдения", "Действия", "Маски", "Решающие". One module replaces SP1's flat float32 codec (`core/action_spec.py`) and the per-seat mask rules (`core/seat_info.py`); both old modules stay untouched until T7.3.

- `ObsSpec.from_space`: `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete` or nested `Dict`; leaves keep the space dtype (`uint8` stays `uint8`); `check` is the cheap structure + shape check of the env contract (dtype is not checked: buffers cast on write).
- `ActionSpec.from_space`: `Discrete`, `MultiDiscrete` (1-D, start 0), 1-D float `Box`, `Units` or nested `Dict`; every non-Dict node is one *group* in natural order.
- Mask trees mirror the discrete parts of the action tree: discrete `bool[n]`, multi_discrete `bool[sum(nvec)]`, box no leaf, units `{"unit": bool[U], "action": bool[U, sum discrete sizes]}`. `normalize_mask` fills missing leaves with "all allowed", rejects unknown keys, wrong shapes and non-bool dtypes (`EnvContractError`) and always returns fresh arrays.
- Empty-row rule (`check_acting_mask`): a non-units discrete group, or one sub-action of a multi_discrete group, with no legal value on an acting seat is an `EnvContractError`; units groups may have empty rows (invalid deciders, handled by `UnitsDist` in T2.2).
- Deciders: `num_deciders = sum(max_units) + (1 if any non-units group)`.

**Files:**
- Create: `src/colosseum/sp2/core/specs.py`
- Test: `tests/unit/test_obs_action_specs.py`

**Interfaces:**
- Consumes: `Tree` (T1.1); `Units`, `UnitComponent` (T1.2); `colosseum.core.errors.EnvContractError`.
- Produces (`colosseum.sp2.core.specs`): `LeafSpec`, `ObsSpec` (`leaves`, `from_space`, `allocate`, `check`, `signature`), `ActionGroup`, `ActionSpec` (`groups`, `is_dict`, `has_units`, `has_masks`, `num_deciders`, `from_space`, `allocate_actions`, `full_mask`, `boot_mask`, `normalize_mask`, `check_acting_mask`, `signature`) — the contract. Additions used by later tasks:
  - `ObsSpec.is_dict: bool` (False: the value is one bare array, leaf path `()`), `ObsSpec.__eq__` / `__hash__` (by signature);
  - `ActionSpec.group_mask(mask, group) -> Any` — the part of a normalized mask tree that belongs to `group` (None for box groups or `mask is None`); `ActionSpec.__eq__` / `__hash__` (by signature);
  - `ObsSpec.from_space` / `ActionSpec.from_space` raise `TypeError` for an unsupported space type and `ValueError` for an unsupported shape/start (an empty `Dict`, a 2-D `Box` action, `Discrete(start=1)`);
  - error messages start with the caller's `where` string.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_obs_action_specs.py`:

```python
"""ObsSpec / ActionSpec v2: leaves, allocation, mask layout, empty-row rules, deciders (SP2 T1.3)."""
import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

from colosseum.core.errors import EnvContractError
from colosseum.sp2.core.specs import ActionSpec, LeafSpec, ObsSpec
from colosseum.sp2.envs.spaces import Units

OBS = Dict([
    ("grid", Box(0, 255, (3, 3), dtype=np.uint8)),
    ("entities", Dict([("feat", Box(-1.0, 1.0, (4, 2))), ("mask", MultiBinary(4))])),
    ("turn", Discrete(9)),
    ("cards", MultiDiscrete([3, 3])),
])

ACT = Dict([
    ("base", Discrete(3)),
    ("aim", Box(-1.0, 1.0, (2,))),
    ("workers", Units(2, Dict([("move", Discrete(4)), ("target", Discrete(3)), ("push", Box(-1.0, 1.0, (1,)))]),
                      only_if={"target": ("move", {3})})),
    ("build", MultiDiscrete([2, 3])),
])


def test_obs_spec_leaves_in_natural_order_with_dtypes():
    spec = ObsSpec.from_space(OBS)
    assert spec.is_dict
    assert spec.leaves == (
        LeafSpec(("grid",), (3, 3), np.dtype(np.uint8)),
        LeafSpec(("entities", "feat"), (4, 2), np.dtype(np.float32)),
        LeafSpec(("entities", "mask"), (4,), np.dtype(np.int8)),
        LeafSpec(("turn",), (), np.dtype(np.int64)),
        LeafSpec(("cards",), (2,), np.dtype(np.int64)),
    )
    assert spec.signature() == ("grid:uint8[3,3];entities/feat:float32[4,2];entities/mask:int8[4];"
                                "turn:int64[];cards:int64[2]")
    bare = ObsSpec.from_space(Box(-1.0, 1.0, (5,)))
    assert not bare.is_dict and bare.leaves == (LeafSpec((), (5,), np.dtype(np.float32)),)


def test_obs_spec_allocate_preserves_dtypes_and_order():
    zeros = ObsSpec.from_space(OBS).allocate((7, 2))
    assert list(zeros) == ["grid", "entities", "turn", "cards"]
    assert zeros["grid"].shape == (7, 2, 3, 3) and zeros["grid"].dtype == np.uint8
    assert zeros["entities"]["mask"].dtype == np.int8 and zeros["turn"].shape == (7, 2)
    assert ObsSpec.from_space(Box(-1.0, 1.0, (5,))).allocate((3,)).shape == (3, 5)


def test_obs_spec_rejects_unsupported_spaces():
    with pytest.raises(TypeError, match="unsupported observation space Tuple"):
        ObsSpec.from_space(Tuple([Discrete(2)]))
    with pytest.raises(ValueError, match="empty Dict"):
        ObsSpec.from_space(Dict([]))


def test_obs_spec_check_structure_and_shapes():
    spec = ObsSpec.from_space(OBS)
    good = OBS.sample()
    spec.check(good, "seat 0: observation")
    with pytest.raises(EnvContractError, match=r"seat 0: observation: keys at <root> do not match.*'cards'"):
        spec.check({k: v for k, v in good.items() if k != "cards"}, "seat 0: observation")
    with pytest.raises(EnvContractError, match=r"leaf entities/feat has shape \(4, 3\), expected \(4, 2\)"):
        spec.check({**good, "entities": {"feat": np.zeros((4, 3)), "mask": good["entities"]["mask"]}}, "obs")
    with pytest.raises(EnvContractError, match="expected a dict"):
        spec.check({**good, "entities": np.zeros(3)}, "obs")
    bare = ObsSpec.from_space(Box(-1.0, 1.0, (5,)))
    bare.check(np.zeros(5, dtype=np.float64), "obs")  # dtype is not checked here (cast on write)
    with pytest.raises(EnvContractError, match="got a dict"):
        bare.check({"x": np.zeros(5)}, "obs")


def test_action_groups_and_deciders():
    spec = ActionSpec.from_space(ACT)
    assert [g.path for g in spec.groups] == [("base",), ("aim",), ("workers",), ("build",)]
    assert [g.kind for g in spec.groups] == ["discrete", "box", "units", "multi_discrete"]
    assert [g.mask_size for g in spec.groups] == [3, 0, 7, 5]
    assert spec.groups[1].box_dim == 2 and spec.groups[3].nvec == (2, 3)
    assert spec.is_dict and spec.has_units and spec.has_masks
    assert spec.num_deciders == 1 + 2      # decider 0 = base+aim+build, then 2 workers
    assert ActionSpec.from_space(Discrete(5)).num_deciders == 1
    assert ActionSpec.from_space(Units(8, Discrete(3))).num_deciders == 8
    two_groups = ActionSpec.from_space(Dict([("a", Units(3, Discrete(2))), ("b", Units(2, Discrete(2)))]))
    assert two_groups.num_deciders == 5 and two_groups.groups[0].path == ("a",)
    box = ActionSpec.from_space(Box(-1.0, 1.0, (3,)))
    assert not box.has_masks and box.full_mask() is None and box.boot_mask() is None


def test_nested_dict_action_space_paths():
    spec = ActionSpec.from_space(Dict([("army", Dict([("stance", Discrete(2)), ("units", Units(2, Discrete(3)))]))]))
    assert [g.path for g in spec.groups] == [("army", "stance"), ("army", "units")]
    mask = spec.full_mask()
    assert list(mask["army"]) == ["stance", "units"] and mask["army"]["units"]["action"].shape == (2, 3)


def test_allocate_actions_native_dtypes():
    acts = ActionSpec.from_space(ACT).allocate_actions((4,))
    assert acts["base"].shape == (4,) and acts["base"].dtype == np.int64
    assert acts["aim"].shape == (4, 2) and acts["aim"].dtype == np.float32
    assert acts["workers"]["move"].shape == (4, 2) and acts["workers"]["push"].shape == (4, 2, 1)
    assert acts["build"].shape == (4, 2) and acts["build"].dtype == np.int64
    assert ActionSpec.from_space(Units(3, MultiDiscrete([2, 5]))).allocate_actions(()).shape == (3, 2)
    assert ActionSpec.from_space(Units(3, Box(-1.0, 1.0, (2,)))).allocate_actions((2,)).dtype == np.float32


def test_full_and_boot_masks():
    spec = ActionSpec.from_space(ACT)
    full = spec.full_mask((2,))
    assert list(full) == ["base", "workers", "build"]          # no leaf for the box group
    assert full["base"].shape == (2, 3) and full["base"].all()
    assert full["workers"]["unit"].shape == (2, 2) and full["workers"]["unit"].all()
    assert full["workers"]["action"].shape == (2, 2, 7) and full["build"].shape == (2, 5)
    boot = spec.boot_mask()
    assert not boot["workers"]["unit"].any() and boot["workers"]["action"].all() and boot["base"].all()


def test_normalize_mask_fills_missing_leaves_and_copies():
    spec = ActionSpec.from_space(ACT)
    base = np.array([True, False, True])
    raw = {"base": base, "workers": {"unit": np.array([True, False])}}
    mask = spec.normalize_mask(raw, "seat 1")
    assert mask["base"].tolist() == [True, False, True] and mask["base"] is not base
    assert mask["workers"]["unit"].tolist() == [True, False]
    assert mask["workers"]["action"].all() and mask["build"].all()
    assert spec.normalize_mask(None, "seat 1")["base"].all()
    flat = ActionSpec.from_space(Discrete(3))
    assert flat.normalize_mask(np.array([False, True, False]), "s").tolist() == [False, True, False]


@pytest.mark.parametrize("raw, message", [
    ({"base": np.array([1, 0, 1], dtype=np.int8)}, "must be a bool array, got dtype int8"),
    ({"base": np.ones(4, dtype=bool)}, r"action mask base has shape \(4,\), expected \(3,\)"),
    ({"aim": np.ones(2, dtype=bool)}, "unexpected key aim"),
    ({"bogus": np.ones(2, dtype=bool)}, "unexpected key bogus"),
    ({"workers": np.ones(2, dtype=bool)}, "must be a dict with keys 'unit' and/or 'action'"),
    ({"workers": {"units": np.ones(2, dtype=bool)}}, "must be a dict with keys 'unit' and/or 'action'"),
    ({"workers": {"action": np.ones((2, 6), dtype=bool)}}, r"workers/action has shape \(2, 6\), expected \(2, 7\)"),
    (np.ones(3, dtype=bool), "must be a dict"),
])
def test_normalize_mask_errors(raw, message):
    with pytest.raises(EnvContractError, match=message):
        ActionSpec.from_space(ACT).normalize_mask(raw, "worker 0, env 1, seat 2")


def test_box_only_action_space_takes_no_mask():
    with pytest.raises(EnvContractError, match="takes no action mask"):
        ActionSpec.from_space(Box(-1.0, 1.0, (2,))).normalize_mask(np.ones(2, dtype=bool), "seat 0")


def test_empty_row_rule_for_acting_seats():
    spec = ActionSpec.from_space(ACT)
    mask = spec.full_mask()
    # Units: an absent unit and an empty component row are allowed (invalid deciders, no error).
    mask["workers"]["unit"][:] = False
    mask["workers"]["action"][0, :4] = False
    spec.check_acting_mask(mask, "seat 0")
    mask["build"][2:] = False                     # sub-action 1 of build (3 values) has no legal value
    with pytest.raises(EnvContractError, match=r"seat 0: action mask build \(sub-action 1\) has no legal action"):
        spec.check_acting_mask(mask, "seat 0")
    flat = ActionSpec.from_space(Discrete(3))
    with pytest.raises(EnvContractError, match="action mask <root> has no legal action"):
        flat.check_acting_mask(np.zeros(3, dtype=bool), "seat 0")
    spec.check_acting_mask(None, "seat 0")


def test_action_spec_rejects_unsupported_spaces():
    with pytest.raises(TypeError, match="unsupported action space"):
        ActionSpec.from_space(Tuple([Discrete(2)]))
    with pytest.raises(ValueError, match="1-D float"):
        ActionSpec.from_space(Box(-1.0, 1.0, (2, 2)))
    with pytest.raises(ValueError, match="start"):
        ActionSpec.from_space(Discrete(3, start=1))


def test_signatures_identify_spaces():
    a = ActionSpec.from_space(ACT)
    assert a == ActionSpec.from_space(ACT)
    assert a.signature().startswith("base:discrete(3);aim:box(2);workers:units(2,dict,")
    assert "target<-move[3]" in a.signature()
    assert ActionSpec.from_space(Discrete(3)) != ActionSpec.from_space(Discrete(4))
    assert ObsSpec.from_space(OBS) == ObsSpec.from_space(OBS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_obs_action_specs.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.core.specs'`.

- [ ] **Step 3: Implement the specs**

Create `src/colosseum/sp2/core/specs.py`:

```python
"""Observation and action specs, and the one set of mask rules (SP2 spec block 2).

Successor of SP1's ``core/action_spec.py`` (flat float32 codec, removed) and
``core/seat_info.py`` (per-seat mask rules). ``EpisodeTracker`` (worker, eval) and
``validate`` use these rules; nothing else re-implements them.

Observations (``ObsSpec``): ``Box``, ``Discrete``, ``MultiBinary``, ``MultiDiscrete`` or a
nested ``Dict`` of them; leaf dtypes are preserved end to end.

Actions (``ActionSpec``): ``Discrete``, ``MultiDiscrete`` (1-D), ``Box`` (1-D), ``Units``
or a nested ``Dict`` of them. Every non-Dict node is one *group*. Action values: discrete
``int64[]``, multi_discrete ``int64[C]``, box ``float32[d]``, units as documented in
:mod:`colosseum.sp2.envs.spaces`; a Dict action is a dict of those.

Mask trees mirror the discrete parts of the action tree; a missing leaf means "all allowed":
discrete ``bool[n]``, multi_discrete ``bool[sum(nvec)]``, box no leaf, units
``{"unit": bool[U], "action": bool[U, sum of discrete component sizes]}``.

Deciders: decider 0 is all non-units groups together (if any), then each units group in
spec order contributes ``max_units`` deciders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium
import numpy as np

from colosseum.core.errors import EnvContractError
from colosseum.sp2.core.tree import Tree
from colosseum.sp2.envs.spaces import Units


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _set_path(tree: dict, path: tuple[str, ...], value: Any) -> None:
    node = tree
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _shape_str(shape: tuple[int, ...]) -> str:
    return "[" + ",".join(str(s) for s in shape) + "]"


# ---------------------------------------------------------------------------
# Observations (and global state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafSpec:
    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype


class ObsSpec:
    """Leaves (path, shape, dtype) of an observation or global-state space."""

    def __init__(self, leaves: tuple[LeafSpec, ...], is_dict: bool) -> None:
        self.leaves = leaves
        self.is_dict = is_dict  # False: the value is one bare array (path ())

    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ObsSpec:
        leaves: list[LeafSpec] = []
        cls._collect(space, (), leaves)
        return cls(tuple(leaves), isinstance(space, gymnasium.spaces.Dict))

    @classmethod
    def _collect(cls, space: gymnasium.Space, path: tuple[str, ...], out: list[LeafSpec]) -> None:
        if isinstance(space, gymnasium.spaces.Dict):
            if not space.spaces:
                raise ValueError(f"observation space at {_fmt(path)}: an empty Dict has no leaves")
            for key, sub in space.spaces.items():
                cls._collect(sub, (*path, key), out)
        elif isinstance(space, gymnasium.spaces.Box | gymnasium.spaces.MultiBinary):
            out.append(LeafSpec(path, tuple(int(s) for s in space.shape), np.dtype(space.dtype)))
        elif isinstance(space, gymnasium.spaces.Discrete):
            out.append(LeafSpec(path, (), np.dtype(space.dtype)))
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            out.append(LeafSpec(path, tuple(int(s) for s in space.nvec.shape), np.dtype(space.dtype)))
        else:
            raise TypeError(f"unsupported observation space {type(space).__name__} at {_fmt(path)}: use Box, "
                            f"Discrete, MultiBinary, MultiDiscrete or a Dict of them")

    def allocate(self, leading: tuple[int, ...]) -> Tree:
        """Zeros with shape ``leading + leaf.shape`` per leaf (numpy, dtypes preserved)."""
        leading = tuple(leading)
        if not self.is_dict:
            leaf = self.leaves[0]
            return np.zeros(leading + leaf.shape, dtype=leaf.dtype)
        out: dict = {}
        for leaf in self.leaves:
            _set_path(out, leaf.path, np.zeros(leading + leaf.shape, dtype=leaf.dtype))
        return out

    def check(self, value: Tree, where: str) -> None:
        """Cheap structure and shape check of one value (no leading dims); EnvContractError."""
        if not self.is_dict:
            if isinstance(value, dict):
                raise EnvContractError(f"{where}: expected an array of shape {self.leaves[0].shape}, got a dict "
                                       f"with keys {list(value)}")
            self._check_leaf(self.leaves[0], value, where)
            return
        self._check_keys(value, (), where)
        for leaf in self.leaves:
            node = value
            for key in leaf.path:
                node = node[key]
            self._check_leaf(leaf, node, where)

    def _check_keys(self, value: Any, path: tuple[str, ...], where: str) -> None:
        expected: list[str] = []
        for leaf in self.leaves:
            if leaf.path[: len(path)] == path and len(leaf.path) > len(path):
                key = leaf.path[len(path)]
                if key not in expected:
                    expected.append(key)
        if not expected:
            return  # a leaf
        if not isinstance(value, dict):
            raise EnvContractError(f"{where}: expected a dict with keys {expected} at {_fmt(path)}, "
                                   f"got {type(value).__name__}")
        if set(value) != set(expected):
            missing = [k for k in expected if k not in value]
            extra = [k for k in value if k not in expected]
            raise EnvContractError(f"{where}: keys at {_fmt(path)} do not match the space "
                                   f"(missing {missing}, unexpected {extra})")
        for key in expected:
            self._check_keys(value[key], (*path, key), where)

    @staticmethod
    def _check_leaf(leaf: LeafSpec, value: Any, where: str) -> None:
        shape = np.shape(value)
        if tuple(shape) != leaf.shape:
            raise EnvContractError(f"{where}: leaf {_fmt(leaf.path)} has shape {tuple(shape)}, "
                                   f"expected {leaf.shape}")

    def signature(self) -> str:
        return ";".join(f"{_fmt(leaf.path)}:{leaf.dtype.name}{_shape_str(leaf.shape)}" for leaf in self.leaves)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObsSpec) and self.signature() == other.signature() and self.is_dict == other.is_dict

    def __hash__(self) -> int:
        return hash(self.signature())

    def __repr__(self) -> str:
        return f"ObsSpec({self.signature()})"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionGroup:
    path: tuple[str, ...]                    # () when the action space is not a Dict
    kind: Literal["discrete", "multi_discrete", "box", "units"]
    nvec: tuple[int, ...]                    # discrete: (n,); multi_discrete: nvec; else ()
    box_dim: int                             # box: d; else 0
    units: Units | None                      # units only
    mask_size: int                           # width of the (per-unit) action mask row


def _units_mask_size(units: Units) -> int:
    return sum(c.size for c in units.components if c.kind == "discrete")


class ActionSpec:
    """Groups of an action space in natural order, with mask and decider rules."""

    def __init__(self, groups: tuple[ActionGroup, ...], is_dict: bool) -> None:
        self.groups = groups
        self.is_dict = is_dict
        self.has_units = any(g.kind == "units" for g in groups)
        self.has_masks = any(g.mask_size > 0 or g.kind == "units" for g in groups)
        has_flat = any(g.kind != "units" for g in groups)
        self.num_deciders = sum(g.units.max_units for g in groups if g.kind == "units") + (1 if has_flat else 0)

    @classmethod
    def from_space(cls, space: gymnasium.Space) -> ActionSpec:
        groups: list[ActionGroup] = []
        cls._collect(space, (), groups)
        return cls(tuple(groups), isinstance(space, gymnasium.spaces.Dict))

    @classmethod
    def _collect(cls, space: gymnasium.Space, path: tuple[str, ...], out: list[ActionGroup]) -> None:
        where = f"action space at {_fmt(path)}"
        if isinstance(space, gymnasium.spaces.Dict):
            if not space.spaces:
                raise ValueError(f"{where}: an empty Dict has no actions")
            for key, sub in space.spaces.items():
                cls._collect(sub, (*path, key), out)
        elif isinstance(space, Units):
            out.append(ActionGroup(path, "units", (), 0, space, _units_mask_size(space)))
        elif isinstance(space, gymnasium.spaces.Discrete):
            if int(space.start) != 0:
                raise ValueError(f"{where}: Discrete(start={int(space.start)}) is not supported; use start=0")
            n = int(space.n)
            out.append(ActionGroup(path, "discrete", (n,), 0, None, n))
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            if space.nvec.ndim != 1 or np.any(space.start != 0):
                raise ValueError(f"{where}: MultiDiscrete must be 1-D with start 0")
            nvec = tuple(int(n) for n in space.nvec)
            out.append(ActionGroup(path, "multi_discrete", nvec, 0, None, sum(nvec)))
        elif isinstance(space, gymnasium.spaces.Box):
            if len(space.shape) != 1 or not np.issubdtype(space.dtype, np.floating):
                raise ValueError(f"{where}: Box actions must be 1-D float, got shape {space.shape} {space.dtype}")
            out.append(ActionGroup(path, "box", (), int(space.shape[0]), None, 0))
        else:
            raise TypeError(f"unsupported {where}: {type(space).__name__} (use Discrete, MultiDiscrete, Box, "
                            f"Units or a Dict of them)")

    # ---- allocation ---------------------------------------------------------

    def _assemble(self, values: dict[tuple[str, ...], Any]) -> Tree:
        if not self.is_dict:
            return values[self.groups[0].path]
        out: dict = {}
        for path, value in values.items():
            _set_path(out, path, value)
        return out

    @staticmethod
    def _zeros_for(group: ActionGroup, leading: tuple[int, ...]) -> Any:
        if group.kind == "discrete":
            return np.zeros(leading, dtype=np.int64)
        if group.kind == "multi_discrete":
            return np.zeros(leading + (len(group.nvec),), dtype=np.int64)
        if group.kind == "box":
            return np.zeros(leading + (group.box_dim,), dtype=np.float32)
        units = group.units
        U = units.max_units
        if units.per_unit_kind == "discrete":
            return np.zeros(leading + (U,), dtype=np.int64)
        if units.per_unit_kind == "multi_discrete":
            return np.zeros(leading + (U, len(units.components)), dtype=np.int64)
        if units.per_unit_kind == "box":
            return np.zeros(leading + (U, units.components[0].size), dtype=np.float32)
        return {c.name: (np.zeros(leading + (U,), dtype=np.int64) if c.kind == "discrete"
                         else np.zeros(leading + (U, c.size), dtype=np.float32)) for c in units.components}

    def allocate_actions(self, leading: tuple[int, ...]) -> Tree:
        """Zero actions with shape ``leading + ...`` (int64 discrete, float32 box)."""
        leading = tuple(leading)
        return self._assemble({g.path: self._zeros_for(g, leading) for g in self.groups})

    def _mask_tree(self, leading: tuple[int, ...], unit_value: bool) -> Tree | None:
        if not self.has_masks:
            return None
        values: dict[tuple[str, ...], Any] = {}
        for g in self.groups:
            if g.kind == "units":
                U = g.units.max_units
                values[g.path] = {"unit": np.full(leading + (U,), unit_value, dtype=bool),
                                  "action": np.ones(leading + (U, g.mask_size), dtype=bool)}
            elif g.mask_size > 0:
                values[g.path] = np.ones(leading + (g.mask_size,), dtype=bool)
        return self._assemble(values)

    def full_mask(self, leading: tuple[int, ...] = ()) -> Tree | None:
        """Everything allowed (units: ``unit=True``); None if the action space has no masks."""
        return self._mask_tree(tuple(leading), True)

    def boot_mask(self) -> Tree | None:
        """The mask of ``boot`` / ``pad`` slots: everything allowed, units ``unit=False``."""
        return self._mask_tree((), False)

    # ---- mask rules -----------------------------------------------------------

    def _raw_group_mask(self, raw: Any, group: ActionGroup, where: str) -> Any:
        if not self.is_dict:
            return raw
        node = raw
        for i, key in enumerate(group.path):
            if not isinstance(node, dict):
                raise EnvContractError(f"{where}: action mask at {_fmt(group.path[:i])} must be a dict, "
                                       f"got {type(node).__name__}")
            if key not in node:
                return None
            node = node[key]
        return node

    def _check_unknown_keys(self, raw: Any, where: str) -> None:
        masked = [g.path for g in self.groups if g.mask_size > 0 or g.kind == "units"]

        def walk(node: Any, path: tuple[str, ...]) -> None:
            if path in masked:
                return
            if not isinstance(node, dict):
                raise EnvContractError(f"{where}: action mask at {_fmt(path)} must be a dict, "
                                       f"got {type(node).__name__}")
            for key, sub in node.items():
                child = (*path, key)
                if not any(m[: len(child)] == child for m in masked):
                    raise EnvContractError(f"{where}: action mask has an unexpected key {_fmt(child)} "
                                           f"(masked groups: {[_fmt(m) for m in masked]})")
                walk(sub, child)

        walk(raw, ())

    @staticmethod
    def _bool_leaf(value: Any, shape: tuple[int, ...], what: str, where: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype != np.bool_:
            raise EnvContractError(f"{where}: {what} must be a bool array, got dtype {arr.dtype}")
        if arr.shape != shape:
            raise EnvContractError(f"{where}: {what} has shape {arr.shape}, expected {shape}")
        return arr

    def normalize_mask(self, raw: Tree | None, where: str) -> Tree | None:
        """A fresh, complete mask tree from an env mask (missing leaves -> all allowed).

        Checks structure, shapes and the bool dtype; EnvContractError otherwise.
        """
        if not self.has_masks:
            if raw is not None:
                raise EnvContractError(f"{where}: the action space has no discrete parts, so it takes no "
                                       f"action mask")
            return None
        out = self.full_mask()
        if raw is None:
            return out
        if self.is_dict:
            self._check_unknown_keys(raw, where)
        for g in self.groups:
            if g.mask_size == 0 and g.kind != "units":
                continue
            part = self._raw_group_mask(raw, g, where)
            if part is None:
                continue
            what = f"action mask {_fmt(g.path)}"
            dst = self.group_mask(out, g)
            if g.kind == "units":
                if not isinstance(part, dict) or not set(part) <= {"unit", "action"}:
                    got = list(part) if isinstance(part, dict) else type(part).__name__
                    raise EnvContractError(f"{where}: {what} of a Units group must be a dict with keys "
                                           f"'unit' and/or 'action', got {got}")
                U = g.units.max_units
                if part.get("unit") is not None:
                    dst["unit"][...] = self._bool_leaf(part["unit"], (U,), f"{what}/unit", where)
                if part.get("action") is not None:
                    dst["action"][...] = self._bool_leaf(part["action"], (U, g.mask_size), f"{what}/action", where)
            else:
                dst[...] = self._bool_leaf(part, (g.mask_size,), what, where)
        return out

    @staticmethod
    def _get(tree: Tree, path: tuple[str, ...]) -> Any:
        node = tree
        for key in path:
            node = node[key]
        return node

    def group_mask(self, mask: Tree | None, group: ActionGroup) -> Any:
        """The part of a normalized mask tree that belongs to ``group`` (None for box / no mask)."""
        if mask is None or (group.mask_size == 0 and group.kind != "units"):
            return None
        return mask if not self.is_dict else self._get(mask, group.path)

    def check_acting_mask(self, mask: Tree | None, where: str) -> None:
        """Empty-row rule for an acting seat: a non-units discrete group (or one sub-action of a
        multi_discrete group) without a legal action is an EnvContractError. Units groups may
        have empty rows (those deciders are invalid)."""
        if mask is None:
            return
        for g in self.groups:
            if g.kind not in ("discrete", "multi_discrete"):
                continue
            row = self.group_mask(mask, g)
            offset = 0
            for i, n in enumerate(g.nvec):
                if not row[offset:offset + n].any():
                    part = "" if g.kind == "discrete" else f" (sub-action {i})"
                    raise EnvContractError(f"{where}: action mask {_fmt(g.path)}{part} has no legal action "
                                           f"for an acting seat")
                offset += n

    # ---- identity -------------------------------------------------------------

    def signature(self) -> str:
        parts = []
        for g in self.groups:
            if g.kind == "discrete":
                body = f"discrete({g.nvec[0]})"
            elif g.kind == "multi_discrete":
                body = f"multi_discrete({','.join(map(str, g.nvec))})"
            elif g.kind == "box":
                body = f"box({g.box_dim})"
            else:
                comps = ",".join(f"{c.name}:{c.kind}({c.size})" for c in g.units.components)
                rules = ",".join(f"{k}<-{p}{sorted(v)}" for k, (p, v) in sorted(g.units.only_if.items()))
                body = f"units({g.units.max_units},{g.units.per_unit_kind},[{comps}],[{rules}])"
            parts.append(f"{_fmt(g.path)}:{body}")
        return ";".join(parts)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActionSpec) and self.signature() == other.signature() and self.is_dict == other.is_dict

    def __hash__(self) -> int:
        return hash(self.signature())

    def __repr__(self) -> str:
        return f"ActionSpec({self.signature()})"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_obs_action_specs.py -q`
Expected: `21 passed`.

- [ ] **Step 5: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/sp2/core/specs.py \
  tests/unit/test_obs_action_specs.py
git commit -m "feat(sp2): ObsSpec, ActionSpec, mask trees, empty-row rule and deciders"
git push origin sp2-game-model
```

---

### Task T1.4: `GameSpec`, `MultiAgentEnv`, `StepResult`, `Outcome`, `resolve_outcome`, toy games

Spec block 1 ("Контракт среды" up to the lifecycle) and the outcome rules of "Конец эпизода".

- `GameSpec(roles, layouts)`: helpers `solo`, `symmetric` (FFA, team = seat, layouts `"<n>p"`), `teams_of` (`"2v2"`, `"2v1v1"`, `"coop<n>"`, seats numbered team by team); `outcome_kind` from the number of teams (1 `score`, 2 `wdl`, 3+ `rank`); `validate` checks roles exist, team numbers are exactly `0..T-1`, at least one layout, safe names (`[A-Za-z0-9_][A-Za-z0-9_-]*`, the agent-id rule without `.`: layout names appear in metric keys and `--layout`), and that every role's spaces are supported by `ObsSpec` / `ActionSpec`. Lists of seats are stored as tuples.
- `resolve_outcome`: default team score = **mean** of the team's seat returns; ranks from scores (`1 + number of strictly better teams`, ties share a rank); `team_rank` only → score is still the mean; keys must be exactly the layout's teams and values finite numbers (`EnvContractError`).
- `tests/game_helpers.py` (new shared kit): the toy games of the overview's "Test support" list, plus `ScriptedGame` (replays prepared results) and `TOY_GAMES` (name -> factory). Every game is pure numpy, deterministic and top-level (picklable for spawn).

**Files:**
- Create: `src/colosseum/sp2/envs/game.py`
- Create: `src/colosseum/sp2/core/outcomes.py`
- Create: `tests/game_helpers.py`
- Modify: `pyproject.toml` (`[tool.ruff.lint.isort] known-first-party`: add `game_helpers`)
- Test: `tests/unit/test_game_spec.py`, `tests/unit/test_team_outcomes.py`, `tests/unit/test_toy_games.py`

**Interfaces:**
- Consumes: `ObsSpec.from_space`, `ActionSpec.from_space` (T1.3, used by `GameSpec.validate`); `Units` (T1.2).
- Produces:
  - `colosseum.sp2.envs.game`: `RoleSpec`, `SeatSpec`, `GameSpec` (`max_seats`, `layout_size`, `teams`, `num_teams`, `outcome_kind`, `role_of`, `validate`, `solo`, `symmetric`, `teams_of`), `MultiAgentEnv`, `Outcome`, `StepResult` — the contract. Addition: `DEFAULT_ROLE = "player"`. Unknown layouts and seats raise `EnvContractError`.
  - `colosseum.sp2.core.outcomes`: `resolve_outcome(outcome, teams, seat_returns, where="")`, `pairwise_rank_score(rank_a, rank_b)` — the contract. Error messages are prefixed with `where` (`"<where>: outcome.team_rank keys ..."`).
  - `tests/game_helpers.py` (imported as `from game_helpers import ...`): `SoloCounterGame(length=8, truncate_at=None)`, `TurnTakingGame(length=6)`, `SimultaneousGame(length=5)`, `EliminationFFA(max_players=4, eliminate_at=None, length=10)`, `TeamDeadTeammateGame(length=6, dead_at=2)`, `UnitsGame(max_units=4, uint8_grid=True, length=6)`, `AsymmetricGame(length=5)`, `CoopGame(size=2, length=5)`, `GlobalStateGame(length=5, truncate_at=None)`, `ScriptedGame(spec, script)` (`received`, `reset_calls`), `TOY_GAMES: dict[str, Callable[[], MultiAgentEnv]]`. The extra keyword arguments with defaults are additions to the contract's signatures (Contract notes). Exact behavior is in each class docstring; later parts rely on it (e.g. `TurnTakingGame` pays the waiting seat, `EliminationFFA` eliminates seat s of an n-seat layout at step n - s by default, `UnitsGame` has `units 0 .. t % U` alive at step t).

- [ ] **Step 1: Write the failing tests**

The toy games are support code for every later SP2 test; their own tests here are smoke tests (spec valid, first results well formed). T1.5 runs every game through the full contract checker.

Create `tests/unit/test_game_spec.py`:

```python
"""GameSpec helpers, outcome kinds and validation (SP2 T1.4)."""
import pickle

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, Tuple

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, RoleSpec, SeatSpec, StepResult

OBS = Box(-1.0, 1.0, (3,), dtype=np.float32)
ACT = Discrete(4)


def test_solo():
    spec = GameSpec.solo(OBS, ACT)
    assert list(spec.roles) == ["player"] and spec.layouts == {"solo": (SeatSpec("player", 0),)}
    assert spec.max_seats == 1 and spec.teams("solo") == [[0]] and spec.outcome_kind("solo") == "score"
    spec.validate()


def test_symmetric_ffa_layouts():
    spec = GameSpec.symmetric(range(2, 5), OBS, ACT)
    assert list(spec.layouts) == ["2p", "3p", "4p"]
    assert spec.layouts["3p"] == (SeatSpec("player", 0), SeatSpec("player", 1), SeatSpec("player", 2))
    assert spec.max_seats == 4 and spec.layout_size("2p") == 2
    assert spec.teams("4p") == [[0], [1], [2], [3]]
    assert spec.outcome_kind("2p") == "wdl" and spec.outcome_kind("3p") == "rank"
    assert list(GameSpec.symmetric(2, OBS, ACT).layouts) == ["2p"]
    with pytest.raises(ValueError, match=">= 1"):
        GameSpec.symmetric(0, OBS, ACT)
    with pytest.raises(ValueError, match="duplicate"):
        GameSpec.symmetric([2, 2], OBS, ACT)


def test_teams_of_names_and_seat_order():
    spec = GameSpec.teams_of([[2, 2], [2, 1, 1], [3]], OBS, ACT, global_state=Box(-1.0, 1.0, (5,)))
    assert list(spec.layouts) == ["2v2", "2v1v1", "coop3"]
    assert [s.team for s in spec.layouts["2v1v1"]] == [0, 0, 1, 2]
    assert spec.teams("2v1v1") == [[0, 1], [2], [3]]
    assert spec.outcome_kind("2v2") == "wdl" and spec.outcome_kind("2v1v1") == "rank"
    assert spec.outcome_kind("coop3") == "score" and spec.num_teams("coop3") == 1
    assert spec.roles["player"].global_state_space is not None
    assert list(GameSpec.teams_of([3, 3], OBS, ACT).layouts) == ["3v3"]
    with pytest.raises(ValueError, match="duplicate"):
        GameSpec.teams_of([[2, 2], [2, 2]], OBS, ACT)


def test_role_of_and_unknown_layout():
    spec = GameSpec(roles={"hunter": RoleSpec(OBS, ACT), "prey": RoleSpec(OBS, Discrete(2))},
                    layouts={"1v2": [SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1)]})
    assert isinstance(spec.layouts["1v2"], tuple)          # lists are stored as tuples
    assert spec.role_of("1v2", 0) == "hunter" and spec.role_of("1v2", 2) == "prey"
    with pytest.raises(EnvContractError, match="seats 0..2, not 3"):
        spec.role_of("1v2", 3)
    with pytest.raises(EnvContractError, match="unknown layout '2v2'"):
        spec.teams("2v2")


@pytest.mark.parametrize("roles, layouts, message", [
    ({}, {"x": (SeatSpec("player", 0),)}, "no roles"),
    ({"player": RoleSpec(OBS, ACT)}, {}, "no layouts"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": ()}, "has no seats"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("ghost", 0),)}, "unknown role 'ghost'"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("player", 0), SeatSpec("player", 2))},
     r"teams \[0, 2\]; team numbers must be exactly 0..T-1"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("player", 1),)}, "team numbers must be exactly"),
    ({"pl.ayer": RoleSpec(OBS, ACT)}, {"x": (SeatSpec("pl.ayer", 0),)}, "role name 'pl.ayer'"),
    ({"player": RoleSpec(OBS, ACT)}, {"2p/x": (SeatSpec("player", 0),)}, "layout name '2p/x'"),
    ({"player": RoleSpec(Tuple([OBS]), ACT)}, {"x": (SeatSpec("player", 0),)}, "role 'player': unsupported"),
    ({"player": RoleSpec(OBS, Box(-1.0, 1.0, (2, 2)))}, {"x": (SeatSpec("player", 0),)}, "1-D float"),
    ({"player": (OBS, ACT)}, {"x": (SeatSpec("player", 0),)}, "must be a RoleSpec"),
    ({"player": RoleSpec(OBS, ACT)}, {"x": (("player", 0),)}, "must be a SeatSpec"),
])
def test_validate_names_the_problem(roles, layouts, message):
    with pytest.raises(EnvContractError, match=message):
        GameSpec(roles=roles, layouts=layouts).validate()


def test_specs_compare_by_value_and_pickle():
    a = GameSpec.symmetric([2, 4], OBS, ACT)
    assert a == GameSpec.symmetric([2, 4], OBS, ACT)
    assert a != GameSpec.symmetric([2, 3], OBS, ACT)
    assert pickle.loads(pickle.dumps(a)) == a


def test_step_result_defaults_and_abstract_env():
    r = StepResult(acting={0}, obs={0: np.zeros(3)})
    assert r.rewards == {} and r.terminated == set() and r.action_masks == {}
    assert not r.episode_over and not r.truncated and r.final_obs is None and r.outcome is None
    with pytest.raises(TypeError):
        MultiAgentEnv()
```

Create `tests/unit/test_team_outcomes.py`:

```python
"""resolve_outcome: default score = mean of seat returns, ranks from scores, checks (SP2 T1.4)."""
import math

import pytest

from colosseum.core.errors import EnvContractError
from colosseum.sp2.core.outcomes import pairwise_rank_score, resolve_outcome
from colosseum.sp2.envs.game import Outcome

TEAMS_2V1V1 = [[0, 1], [2], [3]]


def test_default_score_is_the_mean_of_seat_returns():
    rank, score = resolve_outcome(None, TEAMS_2V1V1, [3.0, 1.0, 2.0, 2.5])
    assert score == {0: 2.0, 1: 2.0, 2: 2.5}          # team 0: (3 + 1) / 2, not 4
    assert rank == {0: 2.0, 1: 2.0, 2: 1.0}           # ties share a rank; 1 + strictly better teams


def test_ranks_from_scores_with_ties():
    rank, _ = resolve_outcome(Outcome(team_score={0: 5.0, 1: 7.0, 2: 5.0, 3: 1.0}),
                              [[0], [1], [2], [3]], [0.0] * 4)
    assert rank == {0: 2.0, 1: 1.0, 2: 2.0, 3: 4.0}


def test_rank_only_keeps_the_mean_score():
    rank, score = resolve_outcome(Outcome(team_rank={0: 2, 1: 1, 2: 2.5}), TEAMS_2V1V1, [1.0, 0.0, 4.0, 9.0])
    assert rank == {0: 2.0, 1: 1.0, 2: 2.5}
    assert score == {0: 0.5, 1: 4.0, 2: 9.0}


def test_both_fields_are_taken_as_given():
    rank, score = resolve_outcome(Outcome(team_rank={0: 1, 1: 2}, team_score={0: -3.0, 1: 10.0}),
                                  [[0], [1]], [0.0, 0.0])
    assert rank == {0: 1.0, 1: 2.0} and score == {0: -3.0, 1: 10.0}


def test_empty_outcome_is_the_default():
    assert resolve_outcome(Outcome(), [[0], [1]], [1.0, 0.0]) == ({0: 1.0, 1: 2.0}, {0: 1.0, 1: 0.0})


def test_single_team_has_rank_one():
    assert resolve_outcome(None, [[0, 1]], [1.0, 3.0]) == ({0: 1.0}, {0: 2.0})


@pytest.mark.parametrize("outcome, message", [
    (Outcome(team_rank={0: 1, 1: 2}), r"team_rank keys \[0, 1\] must be exactly the layout's teams \[0, 1, 2\]"),
    (Outcome(team_score={0: 1.0, 1: 2.0, 2: 3.0, 3: 0.0}), "team_score keys"),
    (Outcome(team_score={0: 1.0, 1: math.nan, 2: 3.0}), r"team_score\[1\] must be finite"),
    (Outcome(team_rank={0: 1, 1: "first", 2: 3}), r"team_rank\[1\] must be a number"),
])
def test_bad_outcomes_raise_with_context(outcome, message):
    with pytest.raises(EnvContractError, match="worker 0, env 2: outcome") as info:
        resolve_outcome(outcome, TEAMS_2V1V1, [0.0] * 4, where="worker 0, env 2")
    assert info.match(message)


def test_pairwise_rank_score():
    assert pairwise_rank_score(1, 2) == 1.0
    assert pairwise_rank_score(2.5, 2.5) == 0.5
    assert pairwise_rank_score(3, 1) == 0.0
```

Create `tests/unit/test_toy_games.py`:

```python
"""The toy games of tests/game_helpers.py have valid specs and well-formed first results (SP2 T1.4)."""
import numpy as np
import pytest

from colosseum.sp2.core.specs import ObsSpec
from colosseum.sp2.envs.game import StepResult
from game_helpers import (
    TOY_GAMES,
    EliminationFFA,
    SoloCounterGame,
    TeamDeadTeammateGame,
    TurnTakingGame,
    UnitsGame,
)


@pytest.mark.parametrize("name", sorted(TOY_GAMES))
def test_every_toy_game_resets_every_layout(name):
    env = TOY_GAMES[name]()
    env.spec.validate()
    for layout in env.spec.layouts:
        result = env.reset(seed=0, layout=layout)
        assert isinstance(result, StepResult) and result.acting and not result.rewards
        for seat in result.acting:
            role = env.spec.roles[env.spec.role_of(layout, seat)]
            ObsSpec.from_space(role.observation_space).check(result.obs[seat], f"{name} seat {seat}")


def test_solo_counter_rewards_and_truncation():
    env = SoloCounterGame(length=8, truncate_at=3)
    env.reset(seed=None, layout="solo")
    assert env.step({0: 1}).rewards == {0: 1.0}
    env.step({0: 0})
    last = env.step({0: 1})
    assert last.episode_over and last.truncated and set(last.final_obs) == {0} and not last.acting


def test_turn_taking_pays_the_waiting_seat():
    env = TurnTakingGame(length=2)
    first = env.reset(seed=None, layout="2p")
    assert first.acting == {0} and first.action_masks[0].tolist() == [True, True, False]
    second = env.step({0: 1})
    assert second.acting == {1} and second.rewards == {1: 1.0}
    assert env.step({1: 0}).episode_over


def test_ffa_default_eliminations_and_ranks():
    env = EliminationFFA(max_players=4)
    r = env.reset(seed=None, layout="4p")
    terminated = []
    while not r.episode_over:
        r = env.step({p: 0 for p in r.acting})
        terminated.append(sorted(r.terminated))
    assert terminated == [[3], [2], [1]]
    assert r.outcome.team_rank == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    assert r.rewards[1] == pytest.approx(-0.9)


def test_dead_teammate_keeps_getting_rewards():
    env = TeamDeadTeammateGame(length=4, dead_at=2)
    r = env.reset(seed=None, layout="2v2")
    for _ in range(2):
        r = env.step({p: 1 for p in r.acting})
    assert r.acting == {0, 2, 3} and r.rewards[1] == 1.0


def test_units_game_births_deaths_and_masks():
    env = UnitsGame(max_units=4)
    r = env.reset(seed=None, layout="solo")
    assert r.obs[0]["grid"].dtype == np.uint8
    mask = r.action_masks[0]["units"]
    assert mask["unit"].tolist() == [True, False, False, False]
    assert not mask["action"][1].any() and mask["action"][0].tolist() == [True] * 4 + [True, False, False, False]
    actions = {"base": 1, "units": {"move": np.zeros(4, np.int64), "target": np.zeros(4, np.int64)}}
    r = env.step({0: actions})
    assert r.rewards == {0: 0.5 + 0.25}
    assert r.action_masks[0]["units"]["unit"].tolist() == [True, True, False, False]
    assert UnitsGame(uint8_grid=False).reset(None, "solo").obs[0]["grid"].dtype == np.float32
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_game_spec.py tests/unit/test_team_outcomes.py tests/unit/test_toy_games.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.envs.game'` (two files) and `No module named 'colosseum.sp2.core.outcomes'`.

- [ ] **Step 3: Implement the env contract types and the outcome rules**

Create `src/colosseum/sp2/envs/game.py`:

```python
"""The SP2 env contract: ``GameSpec`` + ``MultiAgentEnv`` + ``StepResult`` (spec block 1).

A game has *roles* (observation, action and optional global-state spaces) and *layouts*
(match variants): a layout is a tuple of seats, each with a role and a team ``0..T-1``.
Seats ``0..n-1`` of a layout are used; ``n..max_seats-1`` stay empty. The outcome kind
follows from the number of teams: 1 -> ``score``, 2 -> ``wdl``, 3 or more -> ``rank``.

``reset(seed, layout)`` and ``step(actions)`` return a :class:`StepResult`. ``acting`` lists
the seats that act on the NEXT step, and ``step`` receives exactly one action per acting
seat. :class:`colosseum.sp2.envs.contract.EpisodeTracker` checks every rule.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import gymnasium

from colosseum.core.errors import EnvContractError

_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]*")
DEFAULT_ROLE = "player"


@dataclass(frozen=True)
class RoleSpec:
    observation_space: gymnasium.Space
    action_space: gymnasium.Space
    global_state_space: gymnasium.Space | None = None


@dataclass(frozen=True)
class SeatSpec:
    role: str
    team: int


@dataclass(frozen=True)
class GameSpec:
    roles: dict[str, RoleSpec]
    layouts: dict[str, tuple[SeatSpec, ...]]

    def __post_init__(self) -> None:
        # Accept lists of seats; store tuples (the spec is immutable and comparable).
        object.__setattr__(self, "roles", dict(self.roles))
        object.__setattr__(self, "layouts", {name: tuple(seats) for name, seats in self.layouts.items()})

    # ---- structure --------------------------------------------------------

    def _seats(self, layout: str) -> tuple[SeatSpec, ...]:
        try:
            return self.layouts[layout]
        except KeyError:
            raise EnvContractError(f"unknown layout {layout!r}; the game has {list(self.layouts)}") from None

    @property
    def max_seats(self) -> int:
        return max(len(seats) for seats in self.layouts.values())

    def layout_size(self, layout: str) -> int:
        return len(self._seats(layout))

    def teams(self, layout: str) -> list[list[int]]:
        seats = self._seats(layout)
        out: list[list[int]] = [[] for _ in range(1 + max(s.team for s in seats))]
        for i, seat in enumerate(seats):
            out[seat.team].append(i)
        return out

    def num_teams(self, layout: str) -> int:
        return len(self.teams(layout))

    def outcome_kind(self, layout: str) -> Literal["score", "wdl", "rank"]:
        n = self.num_teams(layout)
        return "score" if n == 1 else "wdl" if n == 2 else "rank"

    def role_of(self, layout: str, seat: int) -> str:
        seats = self._seats(layout)
        if not 0 <= seat < len(seats):
            raise EnvContractError(f"layout {layout!r} has seats 0..{len(seats) - 1}, not {seat}")
        return seats[seat].role

    def validate(self) -> None:
        """Raise EnvContractError naming the first problem of the spec."""
        from colosseum.sp2.core.specs import ActionSpec, ObsSpec

        if not self.roles:
            raise EnvContractError("GameSpec: no roles")
        if not self.layouts:
            raise EnvContractError("GameSpec: no layouts (at least one is needed)")
        for name, role in self.roles.items():
            if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
                raise EnvContractError(f"GameSpec: role name {name!r} is not a safe identifier "
                                       f"(letters, digits, '_' and '-', not starting with '-')")
            if not isinstance(role, RoleSpec):
                raise EnvContractError(f"GameSpec: role {name!r} must be a RoleSpec, got {type(role).__name__}")
            try:
                ObsSpec.from_space(role.observation_space)
                ActionSpec.from_space(role.action_space)
                if role.global_state_space is not None:
                    ObsSpec.from_space(role.global_state_space)
            except (TypeError, ValueError) as e:
                raise EnvContractError(f"GameSpec: role {name!r}: {e}") from e
        for layout, seats in self.layouts.items():
            if not isinstance(layout, str) or _NAME_RE.fullmatch(layout) is None:
                raise EnvContractError(f"GameSpec: layout name {layout!r} is not a safe identifier "
                                       f"(letters, digits, '_' and '-', not starting with '-')")
            if not seats:
                raise EnvContractError(f"GameSpec: layout {layout!r} has no seats")
            for i, seat in enumerate(seats):
                if not isinstance(seat, SeatSpec):
                    raise EnvContractError(f"GameSpec: layout {layout!r} seat {i} must be a SeatSpec, "
                                           f"got {type(seat).__name__}")
                if seat.role not in self.roles:
                    raise EnvContractError(f"GameSpec: layout {layout!r} seat {i} has unknown role {seat.role!r} "
                                           f"(roles: {list(self.roles)})")
            teams = sorted({seat.team for seat in seats})
            if teams != list(range(len(teams))):
                raise EnvContractError(f"GameSpec: layout {layout!r} uses teams {teams}; team numbers must be "
                                       f"exactly 0..T-1")

    # ---- helpers -----------------------------------------------------------

    @classmethod
    def solo(cls, obs: gymnasium.Space, act: gymnasium.Space,
             global_state: gymnasium.Space | None = None) -> GameSpec:
        """One seat, role ``"player"``, layout ``"solo"``."""
        return cls(roles={DEFAULT_ROLE: RoleSpec(obs, act, global_state)},
                   layouts={"solo": (SeatSpec(DEFAULT_ROLE, 0),)})

    @classmethod
    def symmetric(cls, num_players: int | Iterable[int], obs: gymnasium.Space, act: gymnasium.Space,
                  global_state: gymnasium.Space | None = None) -> GameSpec:
        """Free-for-all: team = seat; one layout ``"<n>p"`` per player count."""
        counts = [num_players] if isinstance(num_players, int) else list(num_players)
        if not counts or any(int(n) < 1 for n in counts):
            raise ValueError(f"GameSpec.symmetric: player counts must be >= 1, got {counts}")
        layouts = {f"{n}p": tuple(SeatSpec(DEFAULT_ROLE, i) for i in range(int(n))) for n in counts}
        if len(layouts) != len(counts):
            raise ValueError(f"GameSpec.symmetric: duplicate player counts in {counts}")
        return cls(roles={DEFAULT_ROLE: RoleSpec(obs, act, global_state)}, layouts=layouts)

    @classmethod
    def teams_of(cls, sizes: Sequence[int] | Iterable[Sequence[int]], obs: gymnasium.Space, act: gymnasium.Space,
                 global_state: gymnasium.Space | None = None) -> GameSpec:
        """Teams of the given sizes: ``[2, 2]`` -> ``"2v2"``, ``[2, 1, 1]`` -> ``"2v1v1"``, ``[n]`` ->
        ``"coop<n>"``; a list of such lists gives several layouts. Seats are numbered team by team."""
        items = list(sizes)
        groups = [items] if items and all(isinstance(x, int) for x in items) else [list(g) for g in items]
        layouts: dict[str, tuple[SeatSpec, ...]] = {}
        for group in groups:
            if not group or any(int(n) < 1 for n in group):
                raise ValueError(f"GameSpec.teams_of: team sizes must be >= 1, got {group}")
            name = f"coop{group[0]}" if len(group) == 1 else "v".join(str(int(n)) for n in group)
            if name in layouts:
                raise ValueError(f"GameSpec.teams_of: duplicate layout {name!r}")
            layouts[name] = tuple(SeatSpec(DEFAULT_ROLE, t) for t, n in enumerate(group) for _ in range(int(n)))
        return cls(roles={DEFAULT_ROLE: RoleSpec(obs, act, global_state)}, layouts=layouts)


@dataclass
class Outcome:
    """Team-level result of an episode (keys: the layout's teams)."""

    team_rank: dict[int, float] | None = None    # 1 = best; ties share a rank; fractions allowed
    team_score: dict[int, float] | None = None   # game score of the team


@dataclass
class StepResult:
    acting: set[int]                                     # seats that act on the NEXT step
    obs: dict[int, Any]                                  # required for every acting seat
    action_masks: dict[int, Any] = field(default_factory=dict)   # acting seats only; missing = all allowed
    rewards: dict[int, float] = field(default_factory=dict)      # any live seat; missing = 0
    terminated: set[int] = field(default_factory=set)            # seats eliminated in this step
    episode_over: bool = False
    truncated: bool = False                              # ended by an artificial limit
    final_obs: dict[int, Any] | None = None              # with truncated: every live seat
    global_state: dict[int, Any] | None = None           # per seat, from its perspective
    outcome: Outcome | None = None                       # with episode_over
    infos: dict[int, dict] = field(default_factory=dict)


class MultiAgentEnv(ABC):
    """A game with seats; ``spec`` may be a class or an instance attribute."""

    spec: GameSpec

    @abstractmethod
    def reset(self, seed: int | None, layout: str) -> StepResult:
        """Start an episode of ``layout``: acting, obs, action_masks, global_state; no rewards."""

    @abstractmethod
    def step(self, actions: dict[int, Any]) -> StepResult:
        """One step; ``actions`` has exactly the seats of the previous result's ``acting``."""

    def close(self) -> None:
        """Release resources (optional)."""
```

Create `src/colosseum/sp2/core/outcomes.py`:

```python
"""Team ranks and scores of a finished episode (SP2 spec block 1, "Конец эпизода").

- No ``Outcome`` (or neither field set): team score = mean of its seats' episode returns
  (the mean, so teams of different sizes compare fairly), ranks from the scores.
- Only ``team_score``: ranks from the scores (higher is better).
- Only ``team_rank``: score = the same mean of returns.
Ranks from scores: ``rank = 1 + number of strictly better teams`` (ties share a rank).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.game import Outcome


def _ranks_from_scores(score: Mapping[int, float]) -> dict[int, float]:
    return {t: float(1 + sum(1 for u in score if score[u] > s)) for t, s in score.items()}


def _checked(values: Mapping[int, float], field: str, num_teams: int, where: str) -> dict[int, float]:
    prefix = f"{where}: " if where else ""
    if not isinstance(values, Mapping) or set(values) != set(range(num_teams)):
        got = sorted(values) if isinstance(values, Mapping) else type(values).__name__
        raise EnvContractError(f"{prefix}outcome.{field} keys {got} must be exactly the layout's teams "
                               f"{list(range(num_teams))}")
    out: dict[int, float] = {}
    for team in range(num_teams):
        try:
            value = float(values[team])
        except (TypeError, ValueError):
            raise EnvContractError(f"{prefix}outcome.{field}[{team}] must be a number, "
                                   f"got {values[team]!r}") from None
        if not math.isfinite(value):
            raise EnvContractError(f"{prefix}outcome.{field}[{team}] must be finite, got {value}")
        out[team] = value
    return out


def resolve_outcome(outcome: Outcome | None, teams: list[list[int]], seat_returns: Sequence[float],
                    where: str = "") -> tuple[dict[int, float], dict[int, float]]:
    """``(team_rank, team_score)`` for the layout's ``teams`` (team index -> seats)."""
    mean_returns = {t: sum(float(seat_returns[s]) for s in seats) / len(seats) for t, seats in enumerate(teams)}
    rank = score = None
    if outcome is not None:
        if outcome.team_score is not None:
            score = _checked(outcome.team_score, "team_score", len(teams), where)
        if outcome.team_rank is not None:
            rank = _checked(outcome.team_rank, "team_rank", len(teams), where)
    if score is None:
        score = mean_returns
    if rank is None:
        rank = _ranks_from_scores(score)
    return rank, score


def pairwise_rank_score(rank_a: float, rank_b: float) -> float:
    """Score of A against B from team ranks: 1 (A better), 0.5 (tie), 0 (B better)."""
    if rank_a < rank_b:
        return 1.0
    if rank_a == rank_b:
        return 0.5
    return 0.0
```

- [ ] **Step 4: Create the shared test kit with the toy games**

Create `tests/game_helpers.py`:

```python
"""Shared SP2 test kit: toy ``MultiAgentEnv`` games (T1.4), drivers (T1.5), tiny models (T2.3).

Every game is small, pure numpy and deterministic given the reset seed. Games are
top-level classes, so ``functools.partial(Game, ...)`` or the class itself is a picklable
``env_fn`` for spawned processes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary

from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult
from colosseum.sp2.envs.spaces import Units


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


class SoloCounterGame(MultiAgentEnv):
    """Solo, ``length`` steps. Obs ``[t / length, 1]``; ``Discrete(2)``; reward 1 for action 1.

    With ``truncate_at`` (< length) the episode is cut after that many steps (``truncated``,
    ``final_obs``).
    """

    def __init__(self, length: int = 8, truncate_at: int | None = None) -> None:
        self.length, self.truncate_at = length, truncate_at
        self.spec = GameSpec.solo(Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _obs(self) -> np.ndarray:
        return _vec(self.t / self.length, 1.0)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        rewards = {0: float(int(actions[0]) == 1)}
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        if self.truncate_at is not None and self.t >= self.truncate_at:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True, truncated=True,
                              final_obs={0: self._obs()})
        return StepResult(acting={0}, obs={0: self._obs()}, rewards=rewards)


class TurnTakingGame(MultiAgentEnv):
    """Two seats alternate (seat 0 first). Obs ``[t / length, seat, 1]``; ``Discrete(3)`` where action 2
    is always masked. The acting seat's action ``a`` pays ``a`` to the WAITING seat (so seat 1 gets a
    reward before its first move). ``length`` moves in total; outcome from returns (wdl)."""

    def __init__(self, length: int = 6) -> None:
        self.length = length
        self.spec = GameSpec.symmetric(2, Box(-np.inf, np.inf, (3,), dtype=np.float32), Discrete(3))
        self.t = 0

    def _result(self, rewards: dict[int, float]) -> StepResult:
        seat = self.t % 2
        return StepResult(acting={seat}, obs={seat: _vec(self.t / self.length, seat, 1.0)},
                          action_masks={seat: np.array([True, True, False])}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        (seat, action), = actions.items()
        rewards = {1 - seat: float(int(action))}
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class SimultaneousGame(MultiAgentEnv):
    """Matching pennies: both seats act every step for ``length`` steps. Obs ``[t / length, seat]``;
    ``Discrete(2)``; seat 0 gets +1 if the actions match, else -1; seat 1 the opposite."""

    def __init__(self, length: int = 5) -> None:
        self.length = length
        self.spec = GameSpec.symmetric(2, Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: _vec(self.t / self.length, p) for p in (0, 1)}

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return StepResult(acting={0, 1}, obs=self._obs())

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        r = 1.0 if int(actions[0]) == int(actions[1]) else -1.0
        rewards = {0: r, 1: -r}
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return StepResult(acting={0, 1}, obs=self._obs(), rewards=rewards)


class EliminationFFA(MultiAgentEnv):
    """FFA with layouts ``"2p"..f"{max_players}p"``; every live seat acts every step.

    Obs ``[t / length, seat, live seats]``; ``Discrete(2)``. Each acting seat gets 0.1 per step.
    ``eliminate_at`` maps seat -> the step in which it is eliminated (an extra -1 in that step);
    default: in an n-seat layout seat s (s >= 1) is eliminated at step n - s. The episode ends when
    at most one seat is live or after ``length`` steps. Outcome: ``team_rank`` by elimination order
    (survivors share rank 1).
    """

    def __init__(self, max_players: int = 4, eliminate_at: Mapping[int, int] | None = None,
                 length: int = 10) -> None:
        self.max_players, self.length = max_players, length
        self.eliminate_at = dict(eliminate_at) if eliminate_at is not None else None
        self.spec = GameSpec.symmetric(range(2, max_players + 1), Box(-np.inf, np.inf, (3,), dtype=np.float32),
                                       Discrete(2))
        self.n, self.t = 0, 0
        self.live: set[int] = set()
        self.out_step: dict[int, int] = {}
        self.schedule: dict[int, int] = {}

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: _vec(self.t / self.length, p, len(self.live)) for p in sorted(self.live)}

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.n = self.spec.layout_size(layout)
        self.t = 0
        self.live = set(range(self.n))
        self.out_step = {}
        if self.eliminate_at is None:
            self.schedule = {s: self.n - s for s in range(1, self.n)}
        else:
            self.schedule = {s: k for s, k in self.eliminate_at.items() if s < self.n}
        return StepResult(acting=set(self.live), obs=self._obs())

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.t += 1
        rewards = {p: 0.1 for p in actions}
        out = {p for p in self.live if self.schedule.get(p) == self.t}
        for p in out:
            rewards[p] = rewards.get(p, 0.0) - 1.0
            self.out_step[p] = self.t
        self.live -= out
        if len(self.live) <= 1 or self.t >= self.length:
            last = {p: self.out_step.get(p, self.t + 1) for p in range(self.n)}
            rank = {p: float(1 + sum(1 for q in range(self.n) if last[q] > last[p])) for p in range(self.n)}
            return StepResult(acting=set(), obs={}, rewards=rewards, terminated=out, episode_over=True,
                              outcome=Outcome(team_rank=rank))
        return StepResult(acting=set(self.live), obs=self._obs(), rewards=rewards, terminated=out)


class TeamDeadTeammateGame(MultiAgentEnv):
    """``"2v2"`` (seats 0, 1 = team 0; 2, 3 = team 1). Seat 1 stops acting after step ``dead_at`` but is
    not terminated: every step each seat of a team, the dead one included, gets the mean action of the
    team's acting seats. Obs ``[t / length, seat]``; ``Discrete(2)``; ``length`` steps."""

    def __init__(self, length: int = 6, dead_at: int = 2) -> None:
        self.length, self.dead_at = length, dead_at
        self.spec = GameSpec.teams_of([2, 2], Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _acting(self) -> set[int]:
        return {0, 2, 3} if self.t >= self.dead_at else {0, 1, 2, 3}

    def _result(self, rewards: dict[int, float]) -> StepResult:
        acting = self._acting()
        return StepResult(acting=acting, obs={p: _vec(self.t / self.length, p) for p in acting}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        rewards: dict[int, float] = {}
        for team in ((0, 1), (2, 3)):
            moves = [int(actions[p]) for p in team if p in actions]
            for p in team:
                rewards[p] = float(np.mean(moves))
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class UnitsGame(MultiAgentEnv):
    """Solo bot with units. Obs ``Dict(grid uint8 [4, 4] (float32 if not uint8_grid), entities [U, 3],
    entity_mask [U])``. Action ``Dict(base: Discrete(3), units: Units(U, Dict(move: Discrete(4),
    target: Discrete(U)), only_if target <- move == 3))``.

    Units are born and die: at step t, units ``0 .. (t % U)`` exist. Absent units have ``unit=False``
    and empty action rows; a unit may target only existing units. Reward: 0.5 for base action 1 plus
    0.25 per existing unit choosing move 0. ``length`` steps.
    """

    def __init__(self, max_units: int = 4, uint8_grid: bool = True, length: int = 6) -> None:
        self.U, self.uint8_grid, self.length = max_units, uint8_grid, length
        grid = Box(0, 255, (4, 4), dtype=np.uint8) if uint8_grid else Box(0.0, 255.0, (4, 4), dtype=np.float32)
        obs = Dict([("grid", grid), ("entities", Box(-1.0, 1.0, (max_units, 3), dtype=np.float32)),
                    ("entity_mask", MultiBinary(max_units))])
        act = Dict([("base", Discrete(3)),
                    ("units", Units(max_units, Dict([("move", Discrete(4)), ("target", Discrete(max_units))]),
                                    only_if={"target": ("move", {3})}))])
        self.spec = GameSpec.solo(obs, act)
        self.t = 0

    def _alive(self) -> np.ndarray:
        return np.arange(self.U) <= (self.t % self.U)

    def _result(self, rewards: dict[int, float]) -> StepResult:
        alive = self._alive()
        grid = np.full((4, 4), self.t, dtype=np.uint8 if self.uint8_grid else np.float32)
        entities = np.zeros((self.U, 3), dtype=np.float32)
        entities[alive] = [self.t / self.length, 1.0, 0.0]
        obs = {"grid": grid, "entities": entities, "entity_mask": alive.astype(np.int8)}
        action = np.zeros((self.U, 4 + self.U), dtype=bool)
        action[alive, :4] = True
        action[np.ix_(alive, 4 + np.flatnonzero(alive))] = True
        mask = {"units": {"unit": alive.copy(), "action": action}}
        return StepResult(acting={0}, obs={0: obs}, action_masks={0: mask}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        a = actions[0]
        alive = self._alive()
        reward = 0.5 * float(int(a["base"]) == 1) + 0.25 * float(np.sum((np.asarray(a["units"]["move"]) == 0) & alive))
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)
        return self._result({0: reward})


class AsymmetricGame(MultiAgentEnv):
    """Layout ``"1v2"``: seat 0 = role ``hunter`` (obs ``[4]``, ``Discrete(5)``), seats 1, 2 = role ``prey``
    (obs ``[3]``, ``Discrete(3)``), all acting each step for ``length`` steps. The hunter gets 1 per
    prey whose action equals ``hunter_action % 3``; that prey gets -1, the others +0.5."""

    def __init__(self, length: int = 5) -> None:
        self.length = length
        self.spec = GameSpec(
            roles={"hunter": RoleSpec(Box(-np.inf, np.inf, (4,), dtype=np.float32), Discrete(5)),
                   "prey": RoleSpec(Box(-np.inf, np.inf, (3,), dtype=np.float32), Discrete(3))},
            layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))},
        )
        self.t = 0

    def _result(self, rewards: dict[int, float]) -> StepResult:
        f = self.t / self.length
        return StepResult(acting={0, 1, 2}, obs={0: _vec(f, 0, 0, 1), 1: _vec(f, 1, 0), 2: _vec(f, 2, 0)},
                          rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        target = int(actions[0]) % 3
        rewards = {0: 0.0}
        for p in (1, 2):
            caught = int(actions[p]) == target
            rewards[0] += float(caught)
            rewards[p] = -1.0 if caught else 0.5
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class CoopGame(MultiAgentEnv):
    """One team of ``size`` seats (layout ``"coop<size>"``), all acting for ``length`` steps.
    Obs ``[t / length, seat]``; ``Discrete(2)``; every seat gets 1 when all actions are 1."""

    def __init__(self, size: int = 2, length: int = 5) -> None:
        self.size, self.length = size, length
        self.spec = GameSpec.teams_of([size], Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2))
        self.t = 0

    def _result(self, rewards: dict[int, float]) -> StepResult:
        seats = set(range(self.size))
        return StepResult(acting=seats, obs={p: _vec(self.t / self.length, p) for p in seats}, rewards=rewards)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t = 0
        return self._result({})

    def step(self, actions: dict[int, Any]) -> StepResult:
        r = float(all(int(a) == 1 for a in actions.values()))
        rewards = {p: r for p in range(self.size)}
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        return self._result(rewards)


class GlobalStateGame(MultiAgentEnv):
    """``"2p"``, both seats act; obs ``[t / length, seat]``, ``Discrete(2)``, ``global_state`` ``[4]`` =
    ``[seat, t / length, last action of seat 0, last action of seat 1]`` for every acting seat (and,
    with ``truncate_at``, for every live seat with ``final_obs``). Reward: +1 to a seat choosing 1."""

    def __init__(self, length: int = 5, truncate_at: int | None = None) -> None:
        self.length, self.truncate_at = length, truncate_at
        self.spec = GameSpec.symmetric(2, Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(2),
                                       global_state=Box(-np.inf, np.inf, (4,), dtype=np.float32))
        self.t = 0
        self.last = [0, 0]

    def _obs(self) -> dict[int, np.ndarray]:
        return {p: _vec(self.t / self.length, p) for p in (0, 1)}

    def _gs(self) -> dict[int, np.ndarray]:
        return {p: _vec(p, self.t / self.length, *self.last) for p in (0, 1)}

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.t, self.last = 0, [0, 0]
        return StepResult(acting={0, 1}, obs=self._obs(), global_state=self._gs())

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.last = [int(actions[0]), int(actions[1])]
        rewards = {p: float(self.last[p]) for p in (0, 1)}
        self.t += 1
        if self.t >= self.length:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True)
        if self.truncate_at is not None and self.t >= self.truncate_at:
            return StepResult(acting=set(), obs={}, rewards=rewards, episode_over=True, truncated=True,
                              final_obs=self._obs(), global_state=self._gs())
        return StepResult(acting={0, 1}, obs=self._obs(), rewards=rewards, global_state=self._gs())


class ScriptedGame(MultiAgentEnv):
    """Replays prepared results: ``reset`` returns ``script[0]``, the k-th ``step`` returns ``script[k]``.
    The actions it received are kept in ``received``. For contract-violation tests."""

    def __init__(self, spec: GameSpec, script: Sequence[StepResult]) -> None:
        self.spec = spec
        self.script = list(script)
        self.k = 0
        self.received: list[dict[int, Any]] = []
        self.reset_calls: list[tuple[int | None, str]] = []

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self.k = 0
        self.reset_calls.append((seed, layout))
        return self.script[0]

    def step(self, actions: dict[int, Any]) -> StepResult:
        self.received.append(dict(actions))
        self.k += 1
        return self.script[self.k]


TOY_GAMES: dict[str, Callable[[], MultiAgentEnv]] = {
    "solo": SoloCounterGame,
    "turns": TurnTakingGame,
    "simultaneous": SimultaneousGame,
    "ffa": EliminationFFA,
    "dead_teammate": TeamDeadTeammateGame,
    "units": UnitsGame,
    "asymmetric": AsymmetricGame,
    "coop": CoopGame,
    "global_state": GlobalStateGame,
}
```

In `pyproject.toml`, replace:

```toml
known-first-party = ["cli_runner", "colosseum", "dataflow_helpers", "examples", "harness", "helpers", "learning_envs", "ttt_eval"]
```

with:

```toml
known-first-party = ["cli_runner", "colosseum", "dataflow_helpers", "examples", "game_helpers", "harness", "helpers", "learning_envs", "ttt_eval"]
```

Without the `known-first-party` entry ruff's isort treats `game_helpers` as a third-party module and flags the import order of every test that uses it (I001).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_game_spec.py tests/unit/test_team_outcomes.py tests/unit/test_toy_games.py -q`
Expected: `43 passed`.

- [ ] **Step 6: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/sp2/envs/game.py \
  src/colosseum/sp2/core/outcomes.py \
  tests/game_helpers.py \
  pyproject.toml \
  tests/unit/test_game_spec.py \
  tests/unit/test_team_outcomes.py \
  tests/unit/test_toy_games.py
git commit -m "feat(sp2): GameSpec, MultiAgentEnv, StepResult, team outcomes and toy games"
git push origin sp2-game-model
```

---

### Task T1.5: `EpisodeTracker`: seat lifecycle and every contract check

Spec block 1, "Жизненный цикл места", "Шаг без решений", "Конец эпизода", "`global_state`", "`reset`", "Проверки контракта". `EpisodeTracker` checks one env's `StepResult`s against its `GameSpec` and tracks seat phases (EMPTY / LIVE / ELIMINATED), per-seat returns and elimination steps. `MatchRunner` (T3.3, training and eval) and `validate` (T6.2) use it; nothing else re-implements a rule. Every check has its own test:

| Rule | Test |
|---|---|
| reset: no rewards, no `terminated`, no `episode_over`/`truncated`, no outcome | `test_reset_rules` |
| layout not in the spec | `test_unknown_layout` |
| acting seat without an observation | `test_reset_rules` (last case) |
| `actions` keys != previous `acting` | `test_actions_must_be_exactly_the_acting_seats` |
| empty seat in `acting`, or given an observation / mask / reward / `terminated` | `test_empty_seats_get_nothing` |
| reward in the elimination step allowed; rewards afterwards rejected | `test_reward_in_the_elimination_step_is_allowed_and_later_ones_are_not` |
| eliminated seat acting or terminated again | `test_eliminated_seats_cannot_act_or_be_terminated_again` |
| terminated and acting in the same step | `test_terminated_and_acting_in_the_same_step` |
| waiting seats' observations (and masks) are ignored | `test_waiting_seats_observations_and_masks_are_ignored` |
| observation / mask does not fit the role (with the error context) | `test_observation_and_mask_checks_name_the_context` |
| units: empty rows are not an error | `test_units_empty_rows_are_not_an_error` |
| `episode_over` with acting seats; outcome or `truncated` without `episode_over`; non-finite or non-numeric reward | `test_step_result_rules` |
| truncation: `final_obs` for every live seat, none needed for a seat terminated in the same step | `test_truncation_needs_final_obs_for_live_seats_only` |
| `global_state` for acting seats of a role that declares it; final global state at truncation; none for other roles | `test_global_state_rules` |
| more than `max_idle_steps` idle steps in a row | `test_max_idle_steps` |
| outcome keys are exactly the layout's teams (checked in the final step) | `test_team_result_default_and_explicit` |
| dead teammate excluded from `acting` keeps getting rewards and stays LIVE | `test_dead_teammate_keeps_rewards_and_stays_live` |

Errors read `"<context>, seat P, episode step K, layout L: ..."` (parts that do not apply are omitted). `episode_step` counts env steps of the episode (0 at reset, K in the K-th step). `on_step` order: actions check; result checks; rewards applied; `terminated` seats marked ELIMINATED; at `episode_over` the team result is resolved (so a bad `Outcome` is reported in that step).

The shared kit gets two drivers used here and by later parts: `sample_legal_action` (uniform over legal discrete values) and `play_episode` (one episode with random legal actions, every result through a tracker).

**Files:**
- Create: `src/colosseum/sp2/envs/contract.py`
- Modify: `tests/game_helpers.py` (import block; append the drivers section)
- Test: `tests/unit/test_episode_tracker.py`

**Interfaces:**
- Consumes: `GameSpec`, `StepResult`, `Outcome` (T1.4); `resolve_outcome` (T1.4); `ObsSpec.check`, `ActionSpec.normalize_mask`, `ActionSpec.check_acting_mask`, `ActionSpec.group_mask` (T1.3); toy games (T1.4).
- Produces:
  - `colosseum.sp2.envs.contract`: `SeatPhase`, `EpisodeTracker(spec, *, max_idle_steps=1000, context="")` with `layout`, `episode_step`, `episode_over`, `on_reset(layout, result)`, `on_step(actions, result)`, `phase(seat)`, `live_seats()`, `acting()` — the contract. Additions for `MatchRunner` (T3.3) to build `MatchResult`/`EpisodeEnd`:
    - `seat_returns() -> list[float]` (undiscounted return per seat of the layout, rewards of the elimination step included);
    - `eliminated_step(seat) -> int | None`;
    - `team_result() -> tuple[dict[int, float], dict[int, float]]` (`(team_rank, team_score)` resolved in the final step; `RuntimeError` before `episode_over`).
    - `on_step` after `episode_over` and before `on_reset` raise `RuntimeError` (caller bugs, not env errors). `on_step` returns `{}` in the final step.
  - `tests/game_helpers.py`: `sample_legal_action(spec: ActionSpec, mask, rng) -> Tree` (numpy, env format) and `play_episode(env, layout, *, seed=None, rng=None, tracker=None, max_steps=10_000) -> tuple[EpisodeTracker, list[StepResult]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_episode_tracker.py`:

```python
"""EpisodeTracker: every env-contract rule of SP2 spec block 1, seat phases, returns (SP2 T1.5)."""
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.contract import EpisodeTracker, SeatPhase
from colosseum.sp2.envs.game import GameSpec, Outcome, StepResult
from game_helpers import (
    TOY_GAMES,
    EliminationFFA,
    GlobalStateGame,
    SoloCounterGame,
    TeamDeadTeammateGame,
    UnitsGame,
    play_episode,
)

# 2p and 3p layouts: in "2p" seat 2 is empty.
FFA = GameSpec.symmetric([2, 3], Box(-1.0, 1.0, (2,), dtype=np.float32), Discrete(3))
OBS = np.zeros(2, dtype=np.float32)


def _obs(*seats):
    return {s: OBS for s in seats}


def _tracker(layout="3p", acting=(0, 1, 2), **kwargs):
    tracker = EpisodeTracker(FFA, context="worker 0, env 3", **kwargs)
    masks = tracker.on_reset(layout, StepResult(acting=set(acting), obs=_obs(*acting)))
    return tracker, masks


def test_reset_sets_phases_and_returns_full_masks():
    tracker, masks = _tracker("2p", acting=(0, 1))
    assert [tracker.phase(s) for s in range(3)] == [SeatPhase.LIVE, SeatPhase.LIVE, SeatPhase.EMPTY]
    assert tracker.phase(7) is SeatPhase.EMPTY
    assert sorted(masks) == [0, 1] and masks[0].tolist() == [True, True, True]
    assert tracker.acting() == {0, 1} and tracker.live_seats() == [0, 1]
    assert tracker.layout == "2p" and tracker.episode_step == 0 and not tracker.episode_over
    no_masks = EpisodeTracker(GameSpec.solo(Box(-1.0, 1.0, (2,)), Box(-1.0, 1.0, (2,))))
    assert no_masks.on_reset("solo", StepResult(acting={0}, obs={0: np.zeros(2)})) == {0: None}


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0}, obs=_obs(0), rewards={0: 1.0}), "reset must not give rewards"),
    (StepResult(acting={0}, obs=_obs(0), terminated={1}), "reset must not terminate"),
    (StepResult(acting=set(), obs={}, episode_over=True), "reset must not end the episode"),
    (StepResult(acting={0}, obs=_obs(0), outcome=Outcome()), "must not report an outcome"),
    (StepResult(acting={0, 1}, obs=_obs(0)), "seat 1, episode step 0, layout 3p: an acting seat has no observation"),
    ("not a result", "must return a StepResult"),
])
def test_reset_rules(result, message):
    with pytest.raises(EnvContractError, match=message):
        EpisodeTracker(FFA).on_reset("3p", result)


def test_unknown_layout():
    with pytest.raises(EnvContractError, match=r"worker 1, episode step 0: layout '5p' is not one of"):
        EpisodeTracker(FFA, context="worker 1").on_reset("5p", StepResult(acting={0}, obs=_obs(0)))


def test_actions_must_be_exactly_the_acting_seats():
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"actions sent are for seats \[0, 1\], but the acting seats were "
                                               r"\[0, 1, 2\]"):
        tracker.on_step({0: 0, 1: 0}, StepResult(acting={0}, obs=_obs(0)))


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0, 2}, obs=_obs(0, 2)), "seat 2, episode step 1, layout 2p: an empty seat is in acting"),
    (StepResult(acting={0}, obs=_obs(0, 2)), "seat 2.*an observation for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), action_masks={2: np.ones(3, bool)}), "an action mask for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), rewards={2: 1.0}), "seat 2.*a reward for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), terminated={2}), "terminated for an empty seat"),
    (StepResult(acting={0}, obs=_obs(0), rewards={9: 1.0}), "seat 9.*a reward for an empty seat"),
])
def test_empty_seats_get_nothing(result, message):
    tracker, _ = _tracker("2p", acting=(0, 1))
    with pytest.raises(EnvContractError, match=message):
        tracker.on_step({0: 0, 1: 0}, result)


def test_reward_in_the_elimination_step_is_allowed_and_later_ones_are_not():
    tracker, _ = _tracker()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0, 2}, obs=_obs(0, 2), rewards={1: -1.0, 0: 0.5},
                                                   terminated={1}))
    assert tracker.phase(1) is SeatPhase.ELIMINATED and tracker.eliminated_step(1) == 1
    assert tracker.seat_returns() == [0.5, -1.0, 0.0] and tracker.live_seats() == [0, 2]
    act = {0: 0, 2: 0}
    with pytest.raises(EnvContractError, match="seat 1, episode step 2, layout 3p: a reward for a seat eliminated "
                                               "at episode step 1"):
        tracker.on_step(act, StepResult(acting={0, 2}, obs=_obs(0, 2), rewards={1: 0.1}))


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0, 1, 2}, obs=_obs(0, 1, 2)), "seat 1.*a seat eliminated at episode step 1 is in acting"),
    (StepResult(acting={0, 2}, obs=_obs(0, 2), terminated={1}), "terminated for a seat eliminated"),
])
def test_eliminated_seats_cannot_act_or_be_terminated_again(result, message):
    tracker, _ = _tracker()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0, 2}, obs=_obs(0, 2), terminated={1}))
    with pytest.raises(EnvContractError, match=message):
        tracker.on_step({0: 0, 2: 0}, result)


def test_terminated_and_acting_in_the_same_step():
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"seats \[1\] are terminated and acting"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0, 1}, obs=_obs(0, 1), terminated={1}))


def test_waiting_seats_observations_and_masks_are_ignored():
    tracker, _ = _tracker()
    garbage = np.zeros((7, 7))
    masks = tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(
        acting={0}, obs={0: OBS, 1: garbage, 2: garbage},
        action_masks={1: np.zeros(3, dtype=bool)}, rewards={1: 1.0, 2: 2.0}))
    assert list(masks) == [0] and tracker.acting() == {0}
    assert tracker.seat_returns() == [0.0, 1.0, 2.0]


def test_observation_and_mask_checks_name_the_context():
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"^worker 0, env 3, seat 1, episode step 1, layout 3p: observation: "
                                               r"leaf <root> has shape \(3,\), expected \(2,\)"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={1}, obs={1: np.zeros(3)}))
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="seat 0.*action mask: action mask <root> must be a bool array"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0}, obs=_obs(0),
                                                       action_masks={0: np.ones(3, dtype=np.int8)}))
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="seat 0.*has no legal action for an acting seat"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting={0}, obs=_obs(0),
                                                       action_masks={0: np.zeros(3, dtype=bool)}))


def test_units_empty_rows_are_not_an_error():
    env = UnitsGame(max_units=3)
    tracker = EpisodeTracker(env.spec)
    masks = tracker.on_reset("solo", env.reset(None, "solo"))
    assert masks[0]["units"]["unit"].tolist() == [True, False, False]
    assert not masks[0]["units"]["action"][1].any()


@pytest.mark.parametrize("result, message", [
    (StepResult(acting={0}, obs=_obs(0), episode_over=True), r"episode_over with a non-empty acting set \[0\]"),
    (StepResult(acting={0}, obs=_obs(0), outcome=Outcome()), "outcome is only allowed with episode_over"),
    (StepResult(acting={0}, obs=_obs(0), truncated=True), "truncated requires episode_over"),
    (StepResult(acting={0}, obs=_obs(0), rewards={0: float("nan")}), "seat 0.*reward must be finite"),
    (StepResult(acting={0}, obs=_obs(0), rewards={0: "x"}), "reward must be a number"),
])
def test_step_result_rules(result, message):
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=message):
        tracker.on_step({0: 0, 1: 0, 2: 0}, result)


def test_truncation_needs_final_obs_for_live_seats_only():
    # Seat 2 is terminated in the truncation step: it gets terminal, so it needs no final_obs.
    tracker, _ = _tracker()
    end = StepResult(acting=set(), obs={}, terminated={2}, rewards={2: -1.0}, episode_over=True, truncated=True,
                     final_obs=_obs(0, 1))
    assert tracker.on_step({0: 0, 1: 0, 2: 0}, end) == {}
    assert tracker.episode_over and tracker.live_seats() == [0, 1] and tracker.eliminated_step(2) == 1
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="seat 1.*truncated episode without final_obs for this live seat"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True, truncated=True,
                                                       final_obs=_obs(0, 2)))
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match="final_obs: leaf <root> has shape"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True, truncated=True,
                                                       final_obs={0: OBS, 1: OBS, 2: np.zeros(5)}))


def test_episode_end_by_the_rules_needs_no_final_obs():
    tracker, _ = _tracker()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True))
    assert tracker.episode_over and tracker.acting() == set()
    with pytest.raises(RuntimeError, match="after episode_over"):
        tracker.on_step({}, StepResult(acting=set(), obs={}))


def test_global_state_rules():
    spec = GlobalStateGame().spec
    gs = np.zeros(4, dtype=np.float32)
    tracker = EpisodeTracker(spec)
    with pytest.raises(EnvContractError, match="seat 1.*acting seat got no global_state"):
        tracker.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs}))
    tracker = EpisodeTracker(spec)
    tracker.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs, 1: gs}))
    with pytest.raises(EnvContractError, match="seat 0.*truncated episode without the final global_state"):
        tracker.on_step({0: 0, 1: 0}, StepResult(acting=set(), obs={}, episode_over=True, truncated=True,
                                                 final_obs=_obs(0, 1), global_state={1: gs}))
    plain = EpisodeTracker(FFA)
    with pytest.raises(EnvContractError, match="declares no global_state_space"):
        plain.on_reset("2p", StepResult(acting={0, 1}, obs=_obs(0, 1), global_state={0: gs}))


def test_max_idle_steps():
    tracker, _ = _tracker("2p", acting=(0, 1), max_idle_steps=2)
    idle = StepResult(acting=set(), obs={})
    tracker.on_step({0: 0, 1: 0}, idle)
    tracker.on_step({}, idle)
    tracker.on_step({}, StepResult(acting={0}, obs=_obs(0)))   # an acting step resets the count
    tracker.on_step({0: 1}, idle)
    tracker.on_step({}, idle)
    with pytest.raises(EnvContractError, match="more than env.max_idle_steps=2 steps in a row"):
        tracker.on_step({}, idle)


def test_team_result_default_and_explicit():
    tracker, _ = _tracker()
    with pytest.raises(RuntimeError):
        tracker.team_result()
    tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, rewards={0: 1.0, 1: 3.0, 2: 3.0},
                                                   episode_over=True))
    assert tracker.team_result() == ({0: 3.0, 1: 1.0, 2: 1.0}, {0: 1.0, 1: 3.0, 2: 3.0})
    tracker, _ = _tracker()
    with pytest.raises(EnvContractError, match=r"episode step 1, layout 3p: outcome.team_rank keys \[0, 1\]"):
        tracker.on_step({0: 0, 1: 0, 2: 0}, StepResult(acting=set(), obs={}, episode_over=True,
                                                       outcome=Outcome(team_rank={0: 1, 1: 2})))


def test_dead_teammate_keeps_rewards_and_stays_live():
    env = TeamDeadTeammateGame(length=5, dead_at=2)
    tracker, results = play_episode(env, "2v2", rng=np.random.default_rng(1))
    assert all(1 not in r.acting for r in results[2:])
    assert tracker.live_seats() == [0, 1, 2, 3] and tracker.eliminated_step(1) is None
    assert tracker.seat_returns()[1] == pytest.approx(sum(r.rewards.get(1, 0.0) for r in results))


def test_ffa_elimination_bookkeeping():
    tracker, _ = play_episode(EliminationFFA(max_players=4), "4p")
    assert [tracker.eliminated_step(s) for s in range(4)] == [None, 3, 2, 1]
    assert tracker.live_seats() == [0]
    assert tracker.team_result()[0] == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}


@pytest.mark.parametrize("name", sorted(TOY_GAMES))
def test_every_toy_game_satisfies_the_contract(name):
    env = TOY_GAMES[name]()
    rng = np.random.default_rng(0)
    for layout in env.spec.layouts:
        for episode in range(2):
            tracker, results = play_episode(env, layout, seed=episode, rng=rng)
            assert tracker.episode_over and len(results) >= 2


@pytest.mark.parametrize("env", [SoloCounterGame(truncate_at=3), GlobalStateGame(truncate_at=2)],
                         ids=["solo", "global_state"])
def test_truncating_toy_games_satisfy_the_contract(env):
    layout = next(iter(env.spec.layouts))
    tracker, results = play_episode(env, layout)
    assert results[-1].truncated and tracker.live_seats() == sorted(results[-1].final_obs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_episode_tracker.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.envs.contract'`.

- [ ] **Step 3: Implement `EpisodeTracker`**

Create `src/colosseum/sp2/envs/contract.py`:

```python
"""``EpisodeTracker``: checks one env's results against the contract and tracks seat phases.

Every rule of SP2 spec block 1 lives here (with the mask rules of
:mod:`colosseum.sp2.core.specs`); ``MatchRunner`` (training and eval) and ``validate`` use it.

Seat lifecycle (seats ``0..n-1`` of the layout; ``n..max_seats-1`` are EMPTY):
- EMPTY never acts and never gets an observation, a mask, a reward or ``terminated``;
- LIVE acts (in ``acting``) or waits; it may get rewards at any step; observations,
  masks and global states of waiting seats are ignored;
- ELIMINATED (``terminated`` in some step): a reward in that same step is allowed (it is
  applied first); afterwards the seat may not act, get rewards or be terminated again.

A step with an empty ``acting`` and no ``episode_over`` is an idle tick; more than
``max_idle_steps`` of them in a row is an error (an env that forgot ``episode_over``).
"""

from __future__ import annotations

import enum
import math
from typing import Any

from colosseum.core.errors import EnvContractError
from colosseum.sp2.core.outcomes import resolve_outcome
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree
from colosseum.sp2.envs.game import GameSpec, StepResult


class SeatPhase(enum.Enum):
    EMPTY = "empty"
    LIVE = "live"
    ELIMINATED = "eliminated"


class _RoleRules:
    """The specs of one role, built once."""

    def __init__(self, spec: GameSpec, role: str) -> None:
        r = spec.roles[role]
        self.obs = ObsSpec.from_space(r.observation_space)
        self.action = ActionSpec.from_space(r.action_space)
        self.global_state = ObsSpec.from_space(r.global_state_space) if r.global_state_space is not None else None


class EpisodeTracker:
    """Validates one env's StepResults against its GameSpec and tracks seat phases."""

    def __init__(self, spec: GameSpec, *, max_idle_steps: int = 1000, context: str = "") -> None:
        self.spec = spec
        self.max_idle_steps = int(max_idle_steps)
        self.context = context
        self._rules = {role: _RoleRules(spec, role) for role in spec.roles}
        self.layout: str | None = None
        self.episode_step = 0
        self.episode_over = False
        self._phase: list[SeatPhase] = [SeatPhase.EMPTY] * spec.max_seats
        self._acting: set[int] = set()
        self._returns: list[float] = []
        self._eliminated: dict[int, int] = {}
        self._idle = 0
        self._team: tuple[dict[int, float], dict[int, float]] | None = None

    # ---- queries ----------------------------------------------------------------

    def phase(self, seat: int) -> SeatPhase:
        return self._phase[seat] if 0 <= seat < len(self._phase) else SeatPhase.EMPTY

    def live_seats(self) -> list[int]:
        return [s for s, p in enumerate(self._phase) if p is SeatPhase.LIVE]

    def acting(self) -> set[int]:
        return set(self._acting)

    def seat_returns(self) -> list[float]:
        """Undiscounted return of every seat of the layout in this episode so far."""
        return list(self._returns)

    def eliminated_step(self, seat: int) -> int | None:
        """The episode step in which ``seat`` was terminated, or None."""
        return self._eliminated.get(seat)

    def team_result(self) -> tuple[dict[int, float], dict[int, float]]:
        """``(team_rank, team_score)`` of the finished episode (resolved in its final step)."""
        if self._team is None:
            raise RuntimeError("team_result() is only available after episode_over")
        return self._team

    # ---- context ----------------------------------------------------------------

    def _where(self, seat: int | None = None) -> str:
        parts = [self.context] if self.context else []
        if seat is not None:
            parts.append(f"seat {seat}")
        parts.append(f"episode step {self.episode_step}")
        if self.layout is not None:
            parts.append(f"layout {self.layout}")
        return ", ".join(parts)

    def _fail(self, message: str, seat: int | None = None) -> EnvContractError:
        return EnvContractError(f"{self._where(seat)}: {message}")

    def _rules_of(self, seat: int) -> _RoleRules:
        return self._rules[self.spec.role_of(self.layout, seat)]

    # ---- lifecycle -----------------------------------------------------------------

    def on_reset(self, layout: str, result: StepResult) -> dict[int, Tree | None]:
        """Start an episode of ``layout`` with the env's reset result; returns the acting seats' masks."""
        self.layout = None
        self.episode_step = 0
        if layout not in self.spec.layouts:
            raise self._fail(f"layout {layout!r} is not one of the game's layouts {list(self.spec.layouts)}")
        self.layout = layout
        if not isinstance(result, StepResult):
            raise self._fail(f"reset must return a StepResult, got {type(result).__name__}")
        n = self.spec.layout_size(layout)
        self._phase = [SeatPhase.LIVE if s < n else SeatPhase.EMPTY for s in range(self.spec.max_seats)]
        self._returns = [0.0] * n
        self._eliminated = {}
        self._idle = 0
        self.episode_over = False
        self._acting = set()
        self._team = None
        if result.rewards:
            raise self._fail(f"reset must not give rewards, got {result.rewards}")
        if result.terminated:
            raise self._fail(f"reset must not terminate seats, got {sorted(result.terminated)}")
        if result.episode_over or result.truncated:
            raise self._fail("reset must not end the episode (episode_over/truncated set)")
        if result.outcome is not None:
            raise self._fail("reset must not report an outcome")
        return self._accept_next(result)

    def on_step(self, actions: dict[int, Any], result: StepResult) -> dict[int, Tree | None]:
        """Check the actions sent and the env's step result; returns the acting seats' masks."""
        if self.layout is None:
            raise RuntimeError("on_step before on_reset")
        if self.episode_over:
            raise RuntimeError("on_step after episode_over; reset the env first")
        self.episode_step += 1
        if set(actions) != self._acting:
            raise self._fail(f"the actions sent are for seats {sorted(actions)}, but the acting seats were "
                             f"{sorted(self._acting)}")
        if not isinstance(result, StepResult):
            raise self._fail(f"step must return a StepResult, got {type(result).__name__}")
        self._check_seat_keys(result.rewards, "a reward")
        for seat, value in result.rewards.items():
            try:
                reward = float(value)
            except (TypeError, ValueError):
                raise self._fail(f"reward must be a number, got {value!r}", seat) from None
            if not math.isfinite(reward):
                raise self._fail(f"reward must be finite, got {reward}", seat)
        self._check_seat_keys(result.terminated, "terminated")
        if result.outcome is not None and not result.episode_over:
            raise self._fail("outcome is only allowed with episode_over")
        if result.truncated and not result.episode_over:
            raise self._fail("truncated requires episode_over")
        if result.episode_over:
            if result.acting:
                raise self._fail(f"episode_over with a non-empty acting set {sorted(result.acting)}")
            if result.truncated:
                self._check_truncation(result)
        overlap = set(result.acting) & set(result.terminated)
        if overlap:
            raise self._fail(f"seats {sorted(overlap)} are terminated and acting in the same step")
        masks = self._accept_next(result) if not result.episode_over else {}
        for seat, value in result.rewards.items():
            self._returns[seat] += float(value)
        for seat in result.terminated:
            self._phase[seat] = SeatPhase.ELIMINATED
            self._eliminated[seat] = self.episode_step
        if result.episode_over:
            self._team = resolve_outcome(result.outcome, self.spec.teams(self.layout), self._returns, self._where())
            self.episode_over = True
            self._acting = set()
        return masks

    # ---- checks ---------------------------------------------------------------------

    def _check_seat_keys(self, seats: Any, what: str) -> None:
        for seat in seats:
            phase = self.phase(seat)
            if phase is SeatPhase.EMPTY:
                raise self._fail(f"{what} for an empty seat (layout {self.layout} has "
                                 f"{self.spec.layout_size(self.layout)} seats)", seat)
            if phase is SeatPhase.ELIMINATED:
                raise self._fail(f"{what} for a seat eliminated at episode step {self._eliminated[seat]}", seat)

    def _check_truncation(self, result: StepResult) -> None:
        live = [s for s in self.live_seats() if s not in result.terminated]
        final_obs = result.final_obs or {}
        for seat in live:
            if seat not in final_obs:
                raise self._fail("truncated episode without final_obs for this live seat", seat)
            rules = self._rules_of(seat)
            rules.obs.check(final_obs[seat], f"{self._where(seat)}: final_obs")
            if rules.global_state is not None:
                gs = result.global_state or {}
                if seat not in gs:
                    raise self._fail("truncated episode without the final global_state the role declares", seat)
                rules.global_state.check(gs[seat], f"{self._where(seat)}: final global_state")
        self._check_extra_seats(final_obs, "final_obs")
        self._check_extra_seats(result.global_state, "a global_state")
        self._check_global_state_roles(result)

    def _check_global_state_roles(self, result: StepResult) -> None:
        for seat in result.global_state or {}:
            if self._rules_of(seat).global_state is None:
                raise self._fail(f"global_state for a seat whose role {self.spec.role_of(self.layout, seat)!r} "
                                 f"declares no global_state_space", seat)

    def _check_extra_seats(self, per_seat: dict[int, Any] | None, what: str) -> None:
        for seat in per_seat or {}:
            if self.phase(seat) is SeatPhase.EMPTY:
                raise self._fail(f"{what} for an empty seat", seat)

    def _accept_next(self, result: StepResult) -> dict[int, Tree | None]:
        """Check the next acting set with its observations, masks and global states."""
        acting = set(result.acting)
        for seat in acting:
            phase = self.phase(seat)
            if phase is SeatPhase.EMPTY:
                raise self._fail("an empty seat is in acting", seat)
            if phase is SeatPhase.ELIMINATED:
                raise self._fail(f"a seat eliminated at episode step {self._eliminated[seat]} is in acting", seat)
        self._check_extra_seats(result.obs, "an observation")
        self._check_extra_seats(result.action_masks, "an action mask")
        self._check_extra_seats(result.global_state, "a global_state")
        masks: dict[int, Tree | None] = {}
        for seat in sorted(acting):
            rules = self._rules_of(seat)
            where = self._where(seat)
            if seat not in result.obs:
                raise self._fail("an acting seat has no observation", seat)
            rules.obs.check(result.obs[seat], f"{where}: observation")
            gs = result.global_state or {}
            if rules.global_state is not None:
                if seat not in gs:
                    raise self._fail("the role declares a global_state_space but the acting seat got no "
                                     "global_state", seat)
                rules.global_state.check(gs[seat], f"{where}: global_state")
            mask = rules.action.normalize_mask(result.action_masks.get(seat), f"{where}: action mask")
            rules.action.check_acting_mask(mask, where)
            masks[seat] = mask
        self._check_global_state_roles(result)
        if not acting and not result.episode_over:
            self._idle += 1
            if self._idle > self.max_idle_steps:
                raise self._fail(f"more than env.max_idle_steps={self.max_idle_steps} steps in a row without "
                                 f"acting seats and without episode_over")
        elif acting:
            self._idle = 0
        self._acting = acting
        return masks
```

- [ ] **Step 4: Add the drivers to the shared kit**

In `tests/game_helpers.py`, replace:

```python
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult
```

with:

```python
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import Tree
from colosseum.sp2.envs.contract import EpisodeTracker
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult
```

Append to the end of `tests/game_helpers.py`:

```python


# ---------------------------------------------------------------------------
# Drivers (T1.5)
# ---------------------------------------------------------------------------


def _legal(row: np.ndarray | None, n: int) -> np.ndarray:
    legal = np.arange(n) if row is None else np.flatnonzero(row)
    return legal if legal.size else np.zeros(1, dtype=np.int64)


def sample_legal_action(spec: ActionSpec, mask: Tree | None, rng: np.random.Generator) -> Tree:
    """A uniformly random legal action (numpy, env format) for a normalized ``mask`` (or None).

    Discrete parts pick among legal values (0 when a units row is empty or the unit absent);
    box parts are uniform in [-1, 1].
    """
    values: dict[tuple[str, ...], Any] = {}
    for group in spec.groups:
        row = spec.group_mask(mask, group)
        if group.kind == "discrete":
            values[group.path] = np.int64(rng.choice(_legal(row, group.nvec[0])))
        elif group.kind == "multi_discrete":
            out, offset = [], 0
            for n in group.nvec:
                out.append(rng.choice(_legal(None if row is None else row[offset:offset + n], n)))
                offset += n
            values[group.path] = np.array(out, dtype=np.int64)
        elif group.kind == "box":
            values[group.path] = rng.uniform(-1.0, 1.0, group.box_dim).astype(np.float32)
        else:
            units = group.units
            U = units.max_units
            unit = np.ones(U, dtype=bool) if row is None else row["unit"]
            comps: dict[str, np.ndarray] = {}
            offset = 0
            for c in units.components:
                if c.kind == "discrete":
                    arr = np.zeros(U, dtype=np.int64)
                    for u in np.flatnonzero(unit):
                        sub = None if row is None else row["action"][u, offset:offset + c.size]
                        arr[u] = rng.choice(_legal(sub, c.size))
                    offset += c.size
                else:
                    arr = rng.uniform(-1.0, 1.0, (U, c.size)).astype(np.float32)
                    arr[~unit] = 0.0
                comps[c.name] = arr
            if units.per_unit_kind == "dict":
                values[group.path] = comps
            elif units.per_unit_kind == "multi_discrete":
                values[group.path] = np.stack([comps[c.name] for c in units.components], axis=1)
            else:
                values[group.path] = comps["0"]
    if not spec.is_dict:
        return values[spec.groups[0].path]
    out_tree: dict = {}
    for path, value in values.items():
        node = out_tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return out_tree


def play_episode(env: MultiAgentEnv, layout: str, *, seed: int | None = None,
                 rng: np.random.Generator | None = None, tracker: EpisodeTracker | None = None,
                 max_steps: int = 10_000) -> tuple[EpisodeTracker, list[StepResult]]:
    """Play one episode with random legal actions, every result checked by an ``EpisodeTracker``.

    Returns the tracker (phases, returns, team result) and every result (reset first).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    tracker = tracker if tracker is not None else EpisodeTracker(env.spec)
    specs = {role: ActionSpec.from_space(r.action_space) for role, r in env.spec.roles.items()}
    result = env.reset(seed, layout)
    masks = tracker.on_reset(layout, result)
    results = [result]
    for _ in range(max_steps):
        if tracker.episode_over:
            return tracker, results
        actions = {seat: sample_legal_action(specs[env.spec.role_of(layout, seat)], masks[seat], rng)
                   for seat in sorted(tracker.acting())}
        result = env.step(actions)
        masks = tracker.on_step(actions, result)
        results.append(result)
    raise RuntimeError(f"play_episode: no episode_over within {max_steps} steps")
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_episode_tracker.py tests/unit/test_toy_games.py -q`
Expected: `59 passed`.

- [ ] **Step 6: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/sp2/envs/contract.py \
  tests/game_helpers.py \
  tests/unit/test_episode_tracker.py
git commit -m "feat(sp2): EpisodeTracker with seat phases and every env contract check"
git push origin sp2-game-model
```

---

### Task T1.6: `VectorEnv`, `SubprocessVectorEnv` without auto-reset

Spec block 1, "Вектор-среды". Vector envs no longer reset episodes themselves: `step({env: {seat: action}})` steps only the listed envs and returns their results as they are (a finished env returns its final result); `reset({env: (seed, layout)})` resets the listed envs. Every env must report an equal `GameSpec` (validated once). `SubprocessVectorEnv` runs contiguous env slices in spawned children (one torch thread, `OMP_NUM_THREADS=1`, SP1 signal policy via `init_child_process`); each `reset()` / `step()` call is one IPC round with the children that own a requested env, so all resets of a step are batched. Exceptions raised in a child (by the env) are re-raised in the parent with the child traceback attached as a note; every child of the round answers first, so the protocol stays in sync after an error.

**Files:**
- Create: `src/colosseum/sp2/envs/vector.py`
- Test: `tests/unit/test_game_vector_env.py`, `tests/integration/test_game_subproc_vector_env.py`

**Interfaces:**
- Consumes: `GameSpec`, `MultiAgentEnv`, `StepResult` (T1.4); toy games and `ScriptedGame` (T1.4); `colosseum.utils.logging` / `colosseum.utils.process.init_child_process` (SP1, unchanged).
- Produces (`colosseum.sp2.envs.vector`): `VectorEnv(env_fn, num_envs)` and `SubprocessVectorEnv(env_fn, num_envs, num_workers=None)` with `num_envs`, `spec`, `reset(requests)`, `step(actions)`, `close()` — the contract. Additions: `VectorEnv(..., *, index_offset=0)` (global env index in error messages, used by the children), `VectorEnv.envs`, `SubprocessVectorEnv.num_workers`. An unknown layout in `reset` is a `ValueError` (a caller error, raised before the env is touched); an env index out of range is an `IndexError`; `env_fn` returning something that is not a `MultiAgentEnv` with a `GameSpec` is an `EnvContractError`.

- [ ] **Step 1: Write the failing tests**

Env factories passed to `SubprocessVectorEnv` are top-level (picklable under spawn); the integration file starts at most 2 children per vector.

Create `tests/unit/test_game_vector_env.py`:

```python
"""VectorEnv v2: no auto-reset, per-env reset requests, spec equality (SP2 T1.6)."""
import functools

import numpy as np
import pytest

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.vector import VectorEnv
from game_helpers import EliminationFFA, SoloCounterGame


class _ClosingGame(SoloCounterGame):
    closed: list[int] = []

    def close(self) -> None:
        _ClosingGame.closed.append(id(self))


class _OtherLengthSpec(SoloCounterGame):
    """Every second instance has a different observation space."""

    count = 0

    def __init__(self) -> None:
        super().__init__()
        _OtherLengthSpec.count += 1
        if _OtherLengthSpec.count % 2 == 0:
            self.spec = EliminationFFA().spec


def test_reset_and_step_only_the_requested_envs():
    vec = VectorEnv(functools.partial(EliminationFFA, max_players=3), num_envs=3)
    assert vec.num_envs == 3 and list(vec.spec.layouts) == ["2p", "3p"]
    first = vec.reset({0: (1, "2p"), 2: (2, "3p")})
    assert sorted(first) == [0, 2]
    assert first[0].acting == {0, 1} and first[2].acting == {0, 1, 2}
    stepped = vec.step({2: {0: 0, 1: 1, 2: 0}})
    assert list(stepped) == [2] and stepped[2].terminated == {2}
    vec.close()


def test_no_auto_reset_a_finished_env_returns_its_final_result():
    vec = VectorEnv(functools.partial(SoloCounterGame, length=2), num_envs=2)
    vec.reset({0: (None, "solo"), 1: (None, "solo")})
    vec.step({0: {0: 1}, 1: {0: 1}})
    final = vec.step({0: {0: 1}, 1: {0: 0}})
    assert final[0].episode_over and final[1].episode_over and not final[0].acting
    again = vec.reset({1: (None, "solo")})   # only env 1 starts a new episode
    assert list(again) == [1] and again[1].acting == {0}
    vec.close()


def test_unknown_layout_and_bad_index():
    vec = VectorEnv(SoloCounterGame, num_envs=1)
    with pytest.raises(ValueError, match="env 0: unknown layout '2p'"):
        vec.reset({0: (None, "2p")})
    with pytest.raises(IndexError):
        vec.step({3: {0: 1}})
    vec.close()


def test_every_env_must_report_the_same_spec():
    _OtherLengthSpec.count = 0
    with pytest.raises(EnvContractError, match="env 1 reports a different GameSpec than env 0"):
        VectorEnv(_OtherLengthSpec, num_envs=2)


def test_env_fn_must_build_multi_agent_envs():
    with pytest.raises(EnvContractError, match="must return a MultiAgentEnv"):
        VectorEnv(lambda: object(), num_envs=1)


def test_close_closes_every_env():
    _ClosingGame.closed = []
    vec = VectorEnv(_ClosingGame, num_envs=3)
    vec.close()
    assert len(_ClosingGame.closed) == 3
    vec.close()                                # idempotent
    assert len(_ClosingGame.closed) == 3


def test_results_are_numpy():
    vec = VectorEnv(SoloCounterGame, num_envs=1)
    result = vec.reset({0: (None, "solo")})[0]
    assert isinstance(result.obs[0], np.ndarray)
    vec.close()
```

Create `tests/integration/test_game_subproc_vector_env.py`:

```python
"""SubprocessVectorEnv v2: parity with VectorEnv, batched requests, forwarded errors, close (SP2 T1.6)."""
import functools
import time

import numpy as np
import pytest

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.game import StepResult
from colosseum.sp2.envs.vector import SubprocessVectorEnv, VectorEnv
from game_helpers import EliminationFFA, ScriptedGame, SoloCounterGame, UnitsGame


def _ffa():
    return EliminationFFA(max_players=3)


def _short_script():
    """Its first step indexes past the script: an IndexError raised inside the child."""
    first = StepResult(acting={0}, obs={0: np.zeros(2, dtype=np.float32)})
    return ScriptedGame(SoloCounterGame().spec, [first])


class _RaisingGame(SoloCounterGame):
    def step(self, actions):
        raise EnvContractError("bad step from the env")


def _units():
    return UnitsGame(max_units=3)


def _drive(vec, num_steps):
    """Same actions in both vectors; returns the observed (acting, rewards, episode_over) per env and step."""
    trace = []
    vec.reset({e: (e, "3p" if e % 2 else "2p") for e in range(vec.num_envs)})
    acting = {e: ({0, 1, 2} if e % 2 else {0, 1}) for e in range(vec.num_envs)}
    for _ in range(num_steps):
        live = {e: a for e, a in acting.items() if a}
        out = vec.step({e: {p: 0 for p in seats} for e, seats in live.items()})
        trace.append({e: (sorted(r.acting), r.rewards, r.episode_over) for e, r in sorted(out.items())})
        acting = {e: r.acting for e, r in out.items()}
    return trace


def test_parity_with_vector_env():
    ref = VectorEnv(_ffa, num_envs=4)
    sub = SubprocessVectorEnv(_ffa, num_envs=4, num_workers=2)
    try:
        assert sub.spec == ref.spec and sub.num_workers == 2
        assert _drive(sub, 3) == _drive(ref, 3)
    finally:
        ref.close()
        sub.close()


def test_requests_only_reach_the_owning_child_and_keep_dtypes():
    sub = SubprocessVectorEnv(_units, num_envs=3, num_workers=2)   # slices [0, 2) and [2, 3)
    try:
        out = sub.reset({2: (None, "solo")})
        assert list(out) == [2]
        obs = out[2].obs[0]
        assert obs["grid"].dtype == np.uint8 and obs["entity_mask"].dtype == np.int8
        both = sub.reset({0: (None, "solo"), 2: (None, "solo")})   # one round for both children
        assert sorted(both) == [0, 2]
        action = {"base": np.int64(1), "units": {"move": np.zeros(3, np.int64), "target": np.zeros(3, np.int64)}}
        stepped = sub.step({0: {0: action}})
        assert list(stepped) == [0] and stepped[0].rewards == {0: 0.75}
    finally:
        sub.close()


def test_child_errors_are_reraised_with_the_child_traceback():
    sub = SubprocessVectorEnv(_RaisingGame, num_envs=2, num_workers=2)
    try:
        sub.reset({0: (None, "solo"), 1: (None, "solo")})
        with pytest.raises(EnvContractError, match="bad step from the env") as info:
            sub.step({0: {0: 1}, 1: {0: 1}})
        assert any("SubprocessVectorEnv child" in note for note in info.value.__notes__)
        # Both children answered, so the protocol stays in sync: a reset still works.
        assert sub.reset({1: (None, "solo")})[1].acting == {0}
    finally:
        sub.close()


def test_any_env_exception_is_forwarded():
    sub = SubprocessVectorEnv(_short_script, num_envs=1, num_workers=1)
    try:
        sub.reset({0: (None, "solo")})
        with pytest.raises(IndexError):
            sub.step({0: {0: 1}})
    finally:
        sub.close()


def test_unknown_layout_is_rejected_in_the_parent():
    sub = SubprocessVectorEnv(SoloCounterGame, num_envs=1, num_workers=1)
    try:
        with pytest.raises(ValueError, match="unknown layout"):
            sub.reset({0: (None, "4p")})
    finally:
        sub.close()


def test_close_is_bounded_and_idempotent():
    sub = SubprocessVectorEnv(functools.partial(SoloCounterGame, length=3), num_envs=2, num_workers=2)
    procs = list(sub._procs)
    start = time.monotonic()
    sub.close()
    sub.close()
    assert time.monotonic() - start < 10
    assert not any(p.is_alive() for p in procs)


def test_spec_is_checked_in_the_parent_before_spawning():
    with pytest.raises(EnvContractError, match="must be a GameSpec"):
        SubprocessVectorEnv(functools.partial(ScriptedGame, None, []), num_envs=1, num_workers=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_game_vector_env.py tests/integration/test_game_subproc_vector_env.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.envs.vector'`.

- [ ] **Step 3: Implement the vector envs**

Create `src/colosseum/sp2/envs/vector.py`:

```python
"""Vectors of ``MultiAgentEnv``s without auto-reset (SP2 spec block 1, "Вектор-среды").

The caller (``MatchRunner``) decides when and how each env is reset: ``step`` only steps
the envs it is given actions for and returns their results as they are (a finished env
returns its final result); ``reset`` resets the requested envs with ``(seed, layout)``.

``SubprocessVectorEnv`` runs contiguous slices of envs in child processes. Each
``reset()`` / ``step()`` call is one IPC round with the children that own a requested env,
so all resets of a step are batched into one round. Exceptions raised in a child (e.g. by
the env) are re-raised in the parent with the child's traceback as a note.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable, Mapping
from typing import Any

from colosseum.core.errors import EnvContractError
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, StepResult

_CMD_SPEC = "spec"
_CMD_RESET = "reset"
_CMD_STEP = "step"
_CMD_CLOSE = "close"
_JOIN_TIMEOUT = 5.0


def _env_spec(env: Any, index: int) -> GameSpec:
    if not isinstance(env, MultiAgentEnv):
        raise EnvContractError(f"env {index}: env_fn must return a MultiAgentEnv, got {type(env).__name__}")
    spec = getattr(env, "spec", None)
    if not isinstance(spec, GameSpec):
        raise EnvContractError(f"env {index}: {type(env).__name__}.spec must be a GameSpec, "
                               f"got {type(spec).__name__}")
    return spec


class VectorEnv:
    """``num_envs`` envs in this process, stepped one after another."""

    def __init__(self, env_fn: Callable[[], MultiAgentEnv], num_envs: int, *, index_offset: int = 0) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self._offset = index_offset
        self.envs: list[MultiAgentEnv] = []
        try:
            for i in range(num_envs):
                env = env_fn()
                self.envs.append(env)
                spec = _env_spec(env, index_offset + i)
                if i == 0:
                    spec.validate()
                    self.spec: GameSpec = spec
                elif spec != self.spec:
                    raise EnvContractError(f"env {index_offset + i} reports a different GameSpec than env "
                                           f"{index_offset}; every env of a vector must have the same spec")
        except BaseException:
            self.close()
            raise

    def _env(self, index: int) -> MultiAgentEnv:
        if not 0 <= index < self.num_envs:
            raise IndexError(f"env index {index} out of range 0..{self.num_envs - 1}")
        return self.envs[index]

    def reset(self, requests: Mapping[int, tuple[int | None, str]]) -> dict[int, StepResult]:
        """Reset the listed envs: ``{env: (seed, layout)}`` -> ``{env: reset result}``."""
        out: dict[int, StepResult] = {}
        for index, (seed, layout) in requests.items():
            if layout not in self.spec.layouts:
                raise ValueError(f"env {self._offset + index}: unknown layout {layout!r} "
                                 f"(layouts: {list(self.spec.layouts)})")
            out[index] = self._env(index).reset(seed, layout)
        return out

    def step(self, actions: Mapping[int, dict[int, Any]]) -> dict[int, StepResult]:
        """Step the listed envs only: ``{env: {seat: action}}`` -> ``{env: result}``."""
        return {index: self._env(index).step(dict(seat_actions)) for index, seat_actions in actions.items()}

    def close(self) -> None:
        envs, self.envs = self.envs, []
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def _split_indices(num_envs: int, num_workers: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` slices; the first ``num_envs % num_workers`` get one more env."""
    base, extra = divmod(num_envs, num_workers)
    slices, start = [], 0
    for w in range(num_workers):
        size = base + (1 if w < extra else 0)
        if size:
            slices.append((start, start + size))
            start += size
    return slices


def _with_child_traceback(exc: BaseException) -> BaseException:
    exc.add_note("raised in a SubprocessVectorEnv child:\n" + "".join(traceback.format_exception(exc)))
    return exc


def _worker_loop(conn: Any, env_fn: Callable[[], MultiAgentEnv], num_envs: int, offset: int) -> None:
    """Child entry point (top level, so spawn can pickle it): owns envs ``offset .. offset+num_envs-1``."""
    from colosseum.utils.logging import ENV_PROCESS_NAME, inherited_log_dir, setup_process_logging

    setup_process_logging(inherited_log_dir(), f"{os.environ.get(ENV_PROCESS_NAME, 'envproc')}-env{offset}")

    from colosseum.utils.process import init_child_process

    init_child_process()  # Ctrl-C is coordinated by the main process; die with the parent

    import torch

    torch.set_num_threads(1)  # env children only step envs; OMP_NUM_THREADS=1 comes from the parent
    vec: VectorEnv | None = None
    startup_error: BaseException | None = None
    try:
        vec = VectorEnv(env_fn, num_envs, index_offset=offset)
    except Exception as e:  # noqa: BLE001 - reported on the first command
        startup_error = _with_child_traceback(e)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == _CMD_CLOSE:
                conn.send(None)
                break
            if startup_error is not None:
                conn.send(startup_error)
                continue
            try:
                if cmd == _CMD_SPEC:
                    reply: Any = vec.spec
                elif cmd == _CMD_RESET:
                    reply = vec.reset(payload)
                elif cmd == _CMD_STEP:
                    reply = vec.step(payload)
                else:
                    reply = RuntimeError(f"unknown SubprocessVectorEnv command {cmd!r}")
            except Exception as e:  # noqa: BLE001 - forwarded to the parent
                reply = _with_child_traceback(e)
            try:
                conn.send(reply)
            except Exception as e:  # noqa: BLE001 - e.g. an unpicklable exception or result
                conn.send(RuntimeError(f"cannot send the reply of {cmd!r} to the parent: {type(e).__name__}: {e}; "
                                       f"original reply: {reply!r}"))
    except (EOFError, KeyboardInterrupt):
        pass  # the parent went away
    finally:
        if vec is not None:
            try:
                vec.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
        conn.close()


class SubprocessVectorEnv:
    """Same interface as :class:`VectorEnv`; envs live in ``num_workers`` child processes."""

    def __init__(self, env_fn: Callable[[], MultiAgentEnv], num_envs: int, num_workers: int | None = None) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if num_workers is None:
            num_workers = min(num_envs, os.cpu_count() or 1)
        num_workers = max(1, min(num_workers, num_envs))
        self.num_envs = num_envs
        probe = env_fn()  # the spec is read in the parent; children compare theirs with it
        try:
            self.spec: GameSpec = _env_spec(probe, 0)
            self.spec.validate()
        finally:
            probe.close()
        self._slices = _split_indices(num_envs, num_workers)
        self.num_workers = len(self._slices)
        self._owner = [w for w, (start, end) in enumerate(self._slices) for _ in range(start, end)]
        self._ctx = mp.get_context("spawn")
        self._conns: list[Any] = []
        self._procs: list[Any] = []
        self._closed = False
        prev_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"  # children import torch while unpickling env_fn
        try:
            for start, end in self._slices:
                parent_conn, child_conn = self._ctx.Pipe()
                proc = self._ctx.Process(target=_worker_loop, args=(child_conn, env_fn, end - start, start),
                                         daemon=True)
                proc.start()
                child_conn.close()
                self._conns.append(parent_conn)
                self._procs.append(proc)
        finally:
            if prev_omp is None:
                os.environ.pop("OMP_NUM_THREADS", None)
            else:
                os.environ["OMP_NUM_THREADS"] = prev_omp
        try:
            for w, spec in self._round({w: (_CMD_SPEC, None) for w in range(self.num_workers)}).items():
                if spec != self.spec:
                    start = self._slices[w][0]
                    raise EnvContractError(f"env {start} (child {w}) reports a different GameSpec than the env "
                                           f"built in the parent; every env must have the same spec")
        except BaseException:
            self.close()
            raise

    def _round(self, messages: Mapping[int, tuple[str, Any]]) -> dict[int, Any]:
        """Send one message per listed child, then collect every reply (re-raising child errors)."""
        for w, message in messages.items():
            self._conns[w].send(message)
        replies = {w: self._conns[w].recv() for w in messages}
        for reply in replies.values():
            if isinstance(reply, BaseException):
                raise reply
        return replies

    def _by_child(self, per_env: Mapping[int, Any]) -> dict[int, dict[int, Any]]:
        out: dict[int, dict[int, Any]] = {}
        for index, value in per_env.items():
            if not 0 <= index < self.num_envs:
                raise IndexError(f"env index {index} out of range 0..{self.num_envs - 1}")
            w = self._owner[index]
            out.setdefault(w, {})[index - self._slices[w][0]] = value
        return out

    def _merge(self, replies: Mapping[int, dict[int, StepResult]]) -> dict[int, StepResult]:
        return {self._slices[w][0] + local: result for w, part in replies.items() for local, result in part.items()}

    def reset(self, requests: Mapping[int, tuple[int | None, str]]) -> dict[int, StepResult]:
        for index, (_seed, layout) in requests.items():
            if layout not in self.spec.layouts:
                raise ValueError(f"env {index}: unknown layout {layout!r} (layouts: {list(self.spec.layouts)})")
        parts = self._by_child(requests)
        return self._merge(self._round({w: (_CMD_RESET, part) for w, part in parts.items()}))

    def step(self, actions: Mapping[int, dict[int, Any]]) -> dict[int, StepResult]:
        parts = self._by_child(actions)
        return self._merge(self._round({w: (_CMD_STEP, part) for w, part in parts.items()}))

    def close(self) -> None:
        """Close every child (join with a timeout, then terminate/kill); safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send((_CMD_CLOSE, None))
            except (BrokenPipeError, OSError, EOFError):
                pass
        for conn in self._conns:
            try:
                if conn.poll(_JOIN_TIMEOUT):
                    conn.recv()
            except (EOFError, OSError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        for proc in self._procs:
            proc.join(timeout=_JOIN_TIMEOUT)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=_JOIN_TIMEOUT)
            if proc.is_alive():  # pragma: no cover - last resort
                proc.kill()
                proc.join()
        self._conns, self._procs = [], []

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_game_vector_env.py tests/integration/test_game_subproc_vector_env.py -q`
Expected: `14 passed` (10–30 s: the integration file spawns children).

- [ ] **Step 5: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit and push**

```bash
git add src/colosseum/sp2/envs/vector.py \
  tests/unit/test_game_vector_env.py \
  tests/integration/test_game_subproc_vector_env.py
git commit -m "feat(sp2): vector envs with per-env reset requests and no auto-reset"
git push origin sp2-game-model
```

---

### Task T1.7: Config v2

Spec block 9 (config part only; `validate_config` is T6.2) and the config keys of blocks 1, 3, 5, 6, 7. The new schema is a copy of SP1's `core/config.py` under `colosseum.sp2.core.config` with exactly these changes:

- removed: `env.num_players`, `training.phase` (with the `TrainingPhase` enum), the `self_play` section;
- `env.max_idle_steps` (>= 1, default 1000);
- `matchmaking`: `mode` (`self_play` | `league`), `layouts` (weights > 0; empty = every layout equally), `self_play_ratio`, `pfsp_exponent`, `latest_prob`, `teammates` (`self` | `mixed`), `teammate_self_prob`, `shuffle_seats`;
- `checkpoint.interval`, `checkpoint.pool_size` (were `self_play.checkpoint_interval`, `self_play.pool_size`);
- `agents.<id>.roles` (non-empty, no duplicates; existence and equal spaces are checked against the `GameSpec` in T2.4) and `ColosseumConfig.agent_roles(agent_id)`;
- `algorithm.ratio_mode`, `algorithm.unit_trace`, `algorithm.entropy_reduction` (`auto` defaults; their semantics are implemented in T4.2) and the `algorithm_class` default `colosseum.sp2.algorithms.appo.APPO`;
- `networks.critic_encoder_class` (composed models only);
- `rollout.chunk_length >= 2`.

Everything else (`StrictModel` with `extra="forbid"`, agent-id rules and reserved ids, `deep_merge`, `parse_override_value`, `apply_overrides`, `load_config`, `get_agent_config`, `get_trainable_agent_ids`, `config_hash`) keeps its SP1 behavior. `scripts/bench_throughput.py::_make_config` is not touched here: it builds the OLD config until T7.3 switches it (overview, "Shadow package strategy" item 8).

**Files:**
- Create: `src/colosseum/sp2/core/config.py` (copy of `src/colosseum/core/config.py` + the edits below)
- Test: `tests/unit/test_config_v2.py`

**Interfaces:**
- Consumes: `colosseum.core.errors.ConfigError` (SP1).
- Produces (`colosseum.sp2.core.config`): every SP1 name except `TrainingPhase` and `SelfPlayConfig`, plus `MatchmakingConfig`, the new fields of `AlgorithmConfig`, `EnvConfig`, `NetworkConfig`, `RolloutConfig`, `CheckpointConfig`, `AgentOverride`, and `ColosseumConfig.agent_roles(agent_id) -> list[str] | None` — exactly the contract. `agent_roles` raises `ConfigError` for an unknown agent with the same messages as `get_agent_config`. `get_agent_config(agent_id)` still returns a config with `agents == {}` (roles are read from the top-level config).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_config_v2.py`:

```python
"""Config v2: matchmaking, checkpoint keys, roles, unit modes, removed keys, chunk_length >= 2 (SP2 T1.7)."""
import pytest
import yaml
from pydantic import ValidationError

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import (
    AlgorithmConfig,
    CheckpointConfig,
    ColosseumConfig,
    EnvConfig,
    MatchmakingConfig,
    NetworkConfig,
    RolloutConfig,
    TrainingConfig,
    load_config,
    parse_override_value,
)

BASE = {
    "env": {"env_class": "my_game.game.MyGame"},
    "networks": {"encoder_class": "my_game.models.Encoder", "policy_class": "my_game.models.Policy",
                 "value_class": "my_game.models.Value"},
}


def _write(tmp_path, data):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults():
    cfg = ColosseumConfig.model_validate(BASE)
    assert cfg.algorithm.algorithm_class == "colosseum.sp2.algorithms.appo.APPO"
    assert (cfg.algorithm.ratio_mode, cfg.algorithm.unit_trace, cfg.algorithm.entropy_reduction) == \
        ("auto", "auto", "auto")
    assert cfg.env.max_idle_steps == 1000 and cfg.env.kwargs == {}
    assert cfg.networks.critic_encoder_class is None
    m = cfg.matchmaking
    assert (m.mode, m.layouts, m.self_play_ratio, m.pfsp_exponent, m.latest_prob) == ("self_play", {}, 0.5, 1.0, 0.5)
    assert (m.teammates, m.teammate_self_prob, m.shuffle_seats) == ("self", 0.5, True)
    assert (cfg.checkpoint.interval, cfg.checkpoint.pool_size, cfg.checkpoint.save_optimizer) == (1000, 20, True)
    assert cfg.rollout.chunk_length == 256
    assert cfg.get_trainable_agent_ids() == ["agent_0"] and cfg.agent_roles("agent_0") is None


@pytest.mark.parametrize("section, body, key", [
    ("env", {"env_class": "x.Y", "num_players": 2}, "num_players"),
    ("training", {"phase": "self_play"}, "phase"),
    ("self_play", {"pool_size": 5}, "self_play"),
])
def test_removed_keys_are_rejected(tmp_path, section, body, key):
    data = {**BASE, section: body}
    with pytest.raises(ConfigError, match=key):
        load_config(_write(tmp_path, data))


def test_chunk_length_needs_two_slots():
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        RolloutConfig(chunk_length=1)
    assert RolloutConfig(chunk_length=2).chunk_length == 2


@pytest.mark.parametrize("kwargs", [
    {"mode": "arena"}, {"layouts": {"2p": 0.0}}, {"layouts": {"2p": -1.0}}, {"self_play_ratio": 1.5},
    {"latest_prob": -0.1}, {"teammates": "random"}, {"teammate_self_prob": 2.0}, {"pfsp_exponent": -1.0},
])
def test_matchmaking_bounds(kwargs):
    with pytest.raises(ValidationError):
        MatchmakingConfig(**kwargs)


def test_matchmaking_layout_weights():
    m = MatchmakingConfig(mode="league", layouts={"2p": 0.25, "4p": 0.75}, teammates="mixed")
    assert m.layouts == {"2p": 0.25, "4p": 0.75} and m.mode == "league"


def test_checkpoint_bounds():
    with pytest.raises(ValidationError):
        CheckpointConfig(interval=0)
    with pytest.raises(ValidationError):
        CheckpointConfig(pool_size=0)


@pytest.mark.parametrize("field, value", [
    ("ratio_mode", "mean"), ("unit_trace", "sum"), ("entropy_reduction", "mean"),
])
def test_unit_mode_literals(field, value):
    with pytest.raises(ValidationError):
        AlgorithmConfig(**{field: value})
    assert AlgorithmConfig(ratio_mode="per_unit", unit_trace="geo_mean", entropy_reduction="sum").ratio_mode == \
        "per_unit"


def test_env_max_idle_steps_bounds():
    with pytest.raises(ValidationError):
        EnvConfig(env_class="x.Y", max_idle_steps=0)


def test_critic_encoder_is_for_composed_models_only():
    net = NetworkConfig(**BASE["networks"], critic_encoder_class="my_game.models.Critic")
    assert net.critic_encoder_class == "my_game.models.Critic"
    with pytest.raises(ValidationError, match="critic_encoder_class must be omitted"):
        NetworkConfig(model_class="x.Model", critic_encoder_class="x.Critic")


def test_agent_roles():
    cfg = ColosseumConfig.model_validate({**BASE, "agents": {
        "hunter": {"roles": ["hunter"], "algorithm": {"learning_rate": 1e-3}}, "prey": {"roles": ["prey"]},
        "free": None}})
    assert cfg.agent_roles("hunter") == ["hunter"] and cfg.agent_roles("prey") == ["prey"]
    assert cfg.agent_roles("free") is None
    hunter = cfg.get_agent_config("hunter")
    assert hunter.algorithm.learning_rate == 1e-3 and hunter.agents == {}
    with pytest.raises(ConfigError, match="Unknown agent 'ghost'"):
        cfg.agent_roles("ghost")
    with pytest.raises(ConfigError, match="only agent is 'agent_0'"):
        ColosseumConfig.model_validate(BASE).agent_roles("alpha")


@pytest.mark.parametrize("roles, message", [([], "must not be empty"), (["a", "b", "a"], r"\['a'\] more than once")])
def test_bad_agent_roles(roles, message):
    with pytest.raises(ValidationError, match=message):
        ColosseumConfig.model_validate({**BASE, "agents": {"x": {"roles": roles}}})


def test_overrides_reach_the_new_keys(tmp_path):
    overrides = {
        "matchmaking.layouts": parse_override_value("{2p: 1.0}"),
        "matchmaking.layouts.4p": parse_override_value("3"),
        "matchmaking.mode": "league",
        "checkpoint.interval": parse_override_value("50"),
        "env.max_idle_steps": parse_override_value("5"),
        "agents.hunter.roles": parse_override_value("[hunter]"),
    }
    cfg = load_config(_write(tmp_path, BASE), overrides)
    assert cfg.matchmaking.layouts == {"2p": 1.0, "4p": 3.0} and cfg.matchmaking.mode == "league"
    assert cfg.checkpoint.interval == 50 and cfg.env.max_idle_steps == 5
    assert cfg.agent_roles("hunter") == ["hunter"]
    with pytest.raises(ConfigError, match="Unknown config key 'self_play'"):
        load_config(_write(tmp_path, BASE), {"self_play.pool_size": 3})


def test_training_has_no_phase():
    assert "phase" not in TrainingConfig.model_fields
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_config_v2.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.sp2.core.config'`.

- [ ] **Step 3: Copy the SP1 config module**

```bash
cp src/colosseum/core/config.py src/colosseum/sp2/core/config.py
```

- [ ] **Step 4: Apply the config v2 changes**

Each replacement below matches exactly once in the copied file.

In `src/colosseum/sp2/core/config.py` (module docstring), replace:

```python
"""Pydantic v2 configuration models for the Colosseum framework.

Every section of the training pipeline (algorithm, environment, network,
rollout, learner, self-play, checkpointing, metrics, transport, run) has its own
model with sensible defaults.  The top-level :class:`ColosseumConfig` combines
them all and can be loaded from a YAML file via :func:`load_config`.
"""
```

with:

```python
"""Pydantic v2 configuration models for the Colosseum framework (config v2, SP2).

Every section of the training pipeline (algorithm, environment, network,
rollout, learner, matchmaking, checkpointing, metrics, transport, run) has its own
model with sensible defaults.  The top-level :class:`ColosseumConfig` combines
them all and can be loaded from a YAML file via :func:`load_config`.

Changes from SP1: ``env.num_players``, ``training.phase`` and the ``self_play``
section are gone (the game structure comes from the env's ``GameSpec``); new are
``env.max_idle_steps``, ``matchmaking``, ``checkpoint.interval`` / ``pool_size``,
``agents.<id>.roles``, ``algorithm.ratio_mode`` / ``unit_trace`` /
``entropy_reduction`` and ``networks.critic_encoder_class``; ``rollout.chunk_length``
is at least 2.
"""
```

In `src/colosseum/sp2/core/config.py` (the `TrainingPhase` enum is removed: `training.phase` is gone), delete:

```python
class TrainingPhase(str, Enum):
    """High-level training phase."""

    BC = "bc"
    SELF_PLAY = "self_play"
    LEAGUE = "league"


```

In `src/colosseum/sp2/core/config.py` (`AlgorithmConfig.algorithm_class` default: the SP2 APPO of T4.2; T7.3 rewrites `colosseum.sp2`), replace:

```python
        default="colosseum.algorithms.appo.APPO",
```

with:

```python
        default="colosseum.sp2.algorithms.appo.APPO",
```

In `src/colosseum/sp2/core/config.py` (new `AlgorithmConfig` fields after `amp_dtype`), replace:

```python
    use_amp: bool = Field(default=False, description="Enable automatic mixed precision training.")
    amp_dtype: str = Field(default="float16", description="AMP dtype: 'float16' or 'bfloat16'.")
```

with:

```python
    use_amp: bool = Field(default=False, description="Enable automatic mixed precision training.")
    amp_dtype: str = Field(default="float16", description="AMP dtype: 'float16' or 'bfloat16'.")
    ratio_mode: Literal["auto", "joint", "per_unit"] = Field(
        default="auto",
        description="Policy loss over deciders: 'joint' = PPO clip on the joint ratio, advantage times the "
                    "clipped scalar rho; 'per_unit' = ratio and clip per decider with a shared advantage and no "
                    "rho factor, mean over valid deciders. 'auto' = per_unit with Units actions, else joint.",
    )
    unit_trace: Literal["auto", "joint", "geo_mean", "none"] = Field(
        default="auto",
        description="Scalar rho for the V-trace targets: 'joint' = exp(sum of decider log-ratios), 'geo_mean' = "
                    "exp(mean), 'none' = rho = c = 1. 'auto' = joint without Units, geo_mean with Units.",
    )
    entropy_reduction: Literal["auto", "mean_valid", "sum"] = Field(
        default="auto",
        description="How entropy and the kickstart KL are reduced over deciders. 'auto' = sum with "
                    "ratio_mode joint, mean_valid with per_unit.",
    )
```

In `src/colosseum/sp2/core/config.py` (`EnvConfig`: `num_players` removed, `max_idle_steps` added), replace:

```python
        description="Dotted import path to the environment class (e.g. 'examples.tic_tac_toe.env.TicTacToeEnv').",
    )
    num_players: int = Field(default=2, ge=1, description="Number of player slots per match.")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to the env constructor.")
```

with:

```python
        description="Dotted import path to a MultiAgentEnv class (e.g. 'examples.tic_tac_toe.game.TicTacToe').",
    )
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to the env constructor.")
    max_idle_steps: int = Field(
        default=1000, ge=1,
        description="Max steps in a row without acting seats and without episode_over before the env is "
                    "reported as broken (EnvContractError).",
    )
```

In `src/colosseum/sp2/core/config.py` (`NetworkConfig`: new `critic_encoder_class` field), replace:

```python
    value_class: str | None = Field(default=None, description="Dotted path to the value head class.")
    kwargs: dict[str, Any] = Field(
```

with:

```python
    value_class: str | None = Field(default=None, description="Dotted path to the value head class.")
    critic_encoder_class: str | None = Field(
        default=None,
        description="Optional dotted path to a BaseCriticEncoder: the value head then sees core features plus "
                    "the encoded global_state (centralized critic). Composed models only; the agent's role must "
                    "declare a global_state_space.",
    )
    kwargs: dict[str, Any] = Field(
```

In `src/colosseum/sp2/core/config.py` (`NetworkConfig._check_model_spec`: with `model_class`, `critic_encoder_class` must be omitted too), replace:

```python
        parts = {
            "encoder_class": self.encoder_class,
            "core": self.core,
            "policy_class": self.policy_class,
            "value_class": self.value_class,
        }
```

with:

```python
        parts = {
            "encoder_class": self.encoder_class,
            "core": self.core,
            "policy_class": self.policy_class,
            "value_class": self.value_class,
            "critic_encoder_class": self.critic_encoder_class,
        }
```

In `src/colosseum/sp2/core/config.py` (`RolloutConfig.chunk_length`: at least 2 slots), replace:

```python
    chunk_length: int = Field(default=256, ge=1, description="Timesteps per trajectory chunk (T).")
```

with:

```python
    chunk_length: int = Field(
        default=256, ge=2,
        description="Slots per trajectory chunk (S >= 2): decisions (act), bootstrap observations (boot) and "
                    "padding (pad).",
    )
```

In `src/colosseum/sp2/core/config.py` (`TrainingConfig.phase` removed), delete:

```python
    phase: TrainingPhase = Field(default=TrainingPhase.SELF_PLAY, description="Current training phase.")
```

In `src/colosseum/sp2/core/config.py` (the whole `SelfPlayConfig` class, up to `class CheckpointConfig`, becomes `MatchmakingConfig`), replace:

```python
class SelfPlayConfig(StrictModel):
    """Self-play and PFSP / league settings."""

    checkpoint_interval: int = Field(
        default=1000,
        ge=1,
        description="Save a new checkpoint to the self-play pool every N TRAINING steps "
                    "(optimizer updates / policy versions), NOT env steps. Note one training "
                    "step consumes chunk_length * batch_chunks env steps, so pick a value well "
                    "below total_timesteps / (chunk_length * batch_chunks) to actually fill the "
                    "self-play pool.",
    )
    pool_size: int = Field(default=20, ge=1, description="Max checkpoints kept in the FIFO pool per agent.")
    latest_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of sampling the latest policy as an opponent (vs. a historical checkpoint).",
    )
    self_play_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of matches that are solo self-play (rest are arena matches).",
    )
    pfsp_exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="Exponent p in PFSP priority: f(wr) = (1 - wr)^p.",
    )
    shuffle_seats: bool = Field(
        default=True,
        description="Shuffle the seat order of every generated match, so each agent (and each "
                    "checkpoint opponent) plays every seat equally often.",
    )


```

with:

```python
class MatchmakingConfig(StrictModel):
    """How the coordinator builds lineups (layout, match type, team cores, teammates, seats)."""

    mode: Literal["self_play", "league"] = Field(
        default="self_play",
        description="'self_play' = every match is self-play (self_play_ratio is treated as 1); "
                    "'league' = self-play with probability self_play_ratio, else an arena match.",
    )
    layouts: dict[str, float] = Field(
        default_factory=dict,
        description="Layout weights, e.g. {2p: 0.5, 4p: 0.5}. Empty = every layout with equal weight. Only "
                    "layouts with a seat for the data owner's role are drawn.",
    )
    self_play_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Probability of a self-play match in 'league' mode (the rest are arena matches).",
    )
    pfsp_exponent: float = Field(default=1.0, ge=0.0, description="Exponent p in PFSP priority: f(wr) = (1 - wr)^p.")
    latest_prob: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Self-play: probability that an opposing team core is the owner's latest policy "
                    "(else one of its checkpoints).",
    )
    teammates: Literal["self", "mixed"] = Field(
        default="self",
        description="'self' = the team core takes every seat of the team it can play; 'mixed' = each such "
                    "seat goes to the core with probability teammate_self_prob, else to another candidate.",
    )
    teammate_self_prob: float = Field(
        default=0.5, ge=0.0, le=1.0, description="With teammates: mixed, probability that a seat goes to the core.",
    )
    shuffle_seats: bool = Field(
        default=True,
        description="Permute teams with equal role composition and seats of the same role within a team.",
    )

    @field_validator("layouts")
    @classmethod
    def _check_layout_weights(cls, layouts: dict[str, float]) -> dict[str, float]:
        bad = {name: weight for name, weight in layouts.items() if not weight > 0}
        if bad:
            raise ValueError(f"matchmaking.layouts weights must be > 0, got {bad}")
        return layouts


```

In `src/colosseum/sp2/core/config.py` (`CheckpointConfig`: `interval` and `pool_size` moved here from `self_play`), replace:

```python
    """Checkpoint settings. Checkpoints are stored in the run dir (``<run>/checkpoints/``)."""

    save_optimizer
```

with:

```python
    """Checkpoint settings. Checkpoints are stored in the run dir (``<run>/checkpoints/``)."""

    interval: int = Field(
        default=1000, ge=1,
        description="Save a checkpoint every N TRAINING steps (optimizer updates / policy versions), not env "
                    "steps (was self_play.checkpoint_interval).",
    )
    pool_size: int = Field(
        default=20, ge=1, description="Max checkpoints kept in the FIFO pool per agent (was self_play.pool_size).",
    )
    save_optimizer
```

In `src/colosseum/sp2/core/config.py` (`AgentOverride`: new `roles` field), replace:

```python
    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None


_AGENT_SECTIONS
```

with:

```python
    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    roles: list[str] | None = Field(
        default=None,
        description="Roles this agent plays (all must have the same spaces). Omitted = every role of the game, "
                    "which then must all have the same spaces.",
    )

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        if roles is not None:
            if not roles:
                raise ValueError("roles must not be empty (omit it to play every role)")
            duplicates = sorted({r for r in roles if roles.count(r) > 1})
            if duplicates:
                raise ValueError(f"roles lists {duplicates} more than once")
        return roles


_AGENT_SECTIONS
```

In `src/colosseum/sp2/core/config.py` (`ColosseumConfig`: `matchmaking` replaces `self_play`), replace:

```python
    self_play: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
```

with:

```python
    matchmaking: MatchmakingConfig = Field(default_factory=MatchmakingConfig)
```

In `src/colosseum/sp2/core/config.py` (`ColosseumConfig`: new `agent_roles` method before `get_trainable_agent_ids`), replace:

```python
    def get_trainable_agent_ids(self) -> list[str]:
```

with:

```python
    def agent_roles(self, agent_id: str) -> list[str] | None:
        """``agents.<id>.roles`` (None when omitted: the agent plays every role)."""
        if self.agents and agent_id not in self.agents:
            raise ConfigError(f"Unknown agent '{agent_id}'. Known agents: {sorted(self.agents)}")
        if not self.agents and agent_id != "agent_0":
            raise ConfigError(
                f"Unknown agent '{agent_id}': without an 'agents' section the only agent is 'agent_0'"
            )
        override = self.agents.get(agent_id)
        return None if override is None or override.roles is None else list(override.roles)

    def get_trainable_agent_ids(self) -> list[str]:
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_config_v2.py -q`
Expected: `25 passed`.

- [ ] **Step 6: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/sp2/core/config.py \
  tests/unit/test_config_v2.py
git commit -m "feat(sp2): config v2 (matchmaking, checkpoint keys, roles, unit modes)"
git push origin sp2-game-model
```

---

### Task T2.1: Distribution protocol, leaf distributions, `TreeDist`, `make_distribution`

Spec block 2, "«Решающие»", "Протокол распределения", "Встроенные распределения". The protocol is decider-aware: `unit_log_prob`, `unit_entropy`, `unit_valid`, `unit_kl` return `[B, K]` and take the recorded action (with `only_if`, validity depends on it); `log_prob` is the sum of `unit_log_prob` over deciders. Invalid positions are 0 through `torch.where`, never by multiplying with a mask; masked logits use `torch.finfo(dtype).min`, so even an all-illegal row stays finite.

Leaf distributions (`CategoricalDist`, `MultiCategoricalDist`, `DiagGaussianDist`) are one decider each. `TreeDist` composes one distribution per action group of an `ActionSpec`: decider 0 is the sum over all non-units groups, then each units group contributes its `max_units` deciders. `make_distribution(spec, params)` builds the built-in distribution from a parameter tree that mirrors the action tree. Units groups need `UnitsDist`, added in T2.2: until then `make_distribution` raises `NotImplementedError` for them (T2.2 replaces that branch).

The masked categorical entropy keeps SP1's AMP fix: illegal entries contribute exactly 0 to the value and the gradient (`test_masked_entropy_gradient_is_finite_under_loss_scaling`).

**Files:**
- Create: `src/colosseum/sp2/networks/__init__.py` (empty)
- Create: `src/colosseum/sp2/networks/dist/__init__.py`, `base.py`, `leaf.py`, `tree.py`
- Test: `tests/unit/test_leaf_distributions.py`, `tests/unit/test_tree_distribution.py`

**Interfaces:**
- Consumes: `Tree`, `tree_get` (T1.1); `ActionSpec`, `ActionGroup`, `ActionSpec.group_mask` (T1.3).
- Produces (`colosseum.sp2.networks.dist`): `Distribution` (abstract properties `batch_size`, `num_deciders`; `sample`, `mode`, `log_prob`, `unit_log_prob`, `unit_entropy`, `unit_valid`, `unit_kl`, `apply_mask`, `cat`), `CategoricalDist(logits, mask=None)`, `MultiCategoricalDist(logits, nvec, mask=None)`, `DiagGaussianDist(mean, log_std)`, `TreeDist(spec, parts)`, `make_distribution(spec, params)` — the contract. Additions:
  - accessors `CategoricalDist.logits/.mask/.num_categories`, `MultiCategoricalDist.logits/.mask/.nvec`, `DiagGaussianDist.mean/.log_std/.action_dim`, `TreeDist.spec/.parts` (group path -> distribution);
  - module functions in `colosseum.sp2.networks.dist.leaf` reused by `UnitsDist`: `masked_log_softmax`, `categorical_log_prob`, `categorical_entropy`, `categorical_kl`, `sample_categorical`, `gaussian_log_prob`, `gaussian_entropy`, `gaussian_kl`;
  - `apply_mask(None)` returns the same object; `DiagGaussianDist.apply_mask(tensor)` raises `ValueError`; KL between different types or action spaces raises `TypeError`; parameter/shape mismatches raise `ValueError` naming the action group (T6.2 turns them into `ConfigError`).
  - `TreeDist` accepts any `Distribution` for a group whose decider count fits (1 for non-units groups, `max_units` for a units group), so custom per-group distributions plug in; built-in leaf types are also checked for their sizes.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_leaf_distributions.py`:

```python
"""Leaf distributions: exact log-probs, masked entropy without NaN, KL, cat (SP2 T2.1)."""
import math

import pytest
import torch

from colosseum.sp2.networks.dist import CategoricalDist, DiagGaussianDist, MultiCategoricalDist

LOG2, LOG3, LOG4 = math.log(2), math.log(3), math.log(4)


def test_categorical_masked_log_prob_entropy_and_validity():
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    dist = CategoricalDist(torch.zeros(2, 4), mask)
    a = torch.tensor([1, 3])
    assert dist.batch_size == 2 and dist.num_deciders == 1
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2], [-LOG4]]))
    assert torch.allclose(dist.log_prob(a), torch.tensor([-LOG2, -LOG4]))
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[LOG2], [LOG4]]))
    assert dist.unit_valid(a).tolist() == [[True], [True]]


def test_categorical_mode_and_sample_respect_the_mask():
    dist = CategoricalDist(torch.tensor([[5.0, 1.0, 0.0]]), torch.tensor([[False, True, True]]))
    assert dist.mode().tolist() == [1]
    torch.manual_seed(0)
    samples = torch.stack([dist.sample() for _ in range(200)])
    assert samples.dtype == torch.int64 and set(samples.flatten().tolist()) == {1, 2}


def test_masked_entropy_gradient_is_finite_under_loss_scaling():
    logits = torch.randn(3, 5, requires_grad=True)
    mask = torch.tensor([[True, False, True, False, False]] * 3)
    loss = CategoricalDist(logits, mask).unit_entropy(None).sum() * 65536.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[:, [1, 3, 4]] == 0)


def test_all_illegal_row_stays_finite():
    dist = CategoricalDist(torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool))
    assert torch.allclose(dist.unit_log_prob(torch.tensor([2])), torch.tensor([[-LOG4]]))


def test_categorical_kl_exact_and_masked():
    p = CategoricalDist(torch.log(torch.tensor([[0.25, 0.75]])))
    q = CategoricalDist(torch.zeros(1, 2))
    expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
    assert p.unit_kl(q, None).item() == pytest.approx(expected, abs=1e-6)
    assert p.unit_kl(p, None).item() == pytest.approx(0.0, abs=1e-7)
    # A one-sided mask: the KL is taken on the legal intersection, renormalized -> 0 here.
    masked = CategoricalDist(torch.tensor([[3.0, 0.0, 0.0]]), torch.tensor([[False, True, True]]))
    assert masked.unit_kl(CategoricalDist(torch.zeros(1, 3)), None).item() == pytest.approx(0.0, abs=1e-7)
    with pytest.raises(TypeError):
        p.unit_kl(DiagGaussianDist(torch.zeros(1, 2), torch.zeros(2)), None)


def test_categorical_apply_mask_combines_and_cat_fills_missing_masks():
    dist = CategoricalDist(torch.zeros(1, 3), torch.tensor([[True, True, False]]))
    both = dist.apply_mask(torch.tensor([[False, True, True]]))
    assert both.mask.tolist() == [[False, True, False]] and both.mode().tolist() == [1]
    assert dist.apply_mask(None) is dist
    joined = CategoricalDist.cat([dist, CategoricalDist(torch.zeros(2, 3))])
    assert joined.batch_size == 3 and joined.mask.tolist() == [[True, True, False]] + [[True] * 3] * 2


def test_shape_errors():
    with pytest.raises(ValueError, match=r"logits must be \[B, n\]"):
        CategoricalDist(torch.zeros(4))
    with pytest.raises(ValueError, match="mask shape"):
        CategoricalDist(torch.zeros(2, 3), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match=r"\[B, 5\] for nvec \(2, 3\)"):
        MultiCategoricalDist(torch.zeros(2, 4), [2, 3])


def test_multi_categorical_is_one_decider():
    mask = torch.tensor([[True, True, True, False, True]])           # nvec (2, 3): second part [T, F, T]
    dist = MultiCategoricalDist(torch.zeros(1, 5), [2, 3], mask)
    a = torch.tensor([[1, 2]])
    assert dist.num_deciders == 1
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-2 * LOG2]]))
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[2 * LOG2]]))
    sample = dist.sample()
    assert sample.shape == (1, 2) and sample.dtype == torch.int64 and sample[0, 1].item() in (0, 2)
    assert dist.mode().shape == (1, 2)
    assert dist.unit_kl(dist, a).item() == pytest.approx(0.0, abs=1e-7)
    assert MultiCategoricalDist.cat([dist, dist]).batch_size == 2


def test_gaussian_exact_values():
    dist = DiagGaussianDist(torch.zeros(3, 2), torch.zeros(2))            # log_std broadcast from [d]
    a = torch.zeros(3, 2)
    assert torch.allclose(dist.unit_log_prob(a), torch.full((3, 1), -math.log(2 * math.pi)))
    assert torch.allclose(dist.unit_entropy(a), torch.full((3, 1), 1.0 + math.log(2 * math.pi)))
    other = DiagGaussianDist(torch.ones(3, 2), torch.zeros(3, 2))
    assert torch.allclose(dist.unit_kl(other, a), torch.full((3, 1), 1.0))   # 0.5 per dim
    assert dist.mode().tolist() == [[0.0, 0.0]] * 3 and dist.sample().shape == (3, 2)
    assert dist.apply_mask(None) is dist
    with pytest.raises(ValueError, match="takes no mask"):
        dist.apply_mask(torch.ones(3, 2, dtype=torch.bool))
    joined = DiagGaussianDist.cat([dist, other])
    assert joined.batch_size == 6 and joined.log_std.shape == (6, 2)


def test_gaussian_log_prob_matches_torch():
    mean, log_std = torch.randn(4, 3), torch.randn(4, 3) * 0.3
    a = torch.randn(4, 3)
    ref = torch.distributions.Normal(mean, log_std.exp()).log_prob(a).sum(-1)
    assert torch.allclose(DiagGaussianDist(mean, log_std).log_prob(a), ref, atol=1e-5)
```

Create `tests/unit/test_tree_distribution.py`:

```python
"""TreeDist and make_distribution over Dict action spaces (SP2 T2.1)."""
import math

import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.networks.dist import CategoricalDist, TreeDist, make_distribution

SPEC = ActionSpec.from_space(Dict([("move", Discrete(3)), ("aim", Box(-1.0, 1.0, (2,))),
                                   ("build", MultiDiscrete([2, 2]))]))


def _params(batch=2):
    return {"move": torch.zeros(batch, 3), "aim": {"mean": torch.zeros(batch, 2), "log_std": torch.zeros(2)},
            "build": torch.zeros(batch, 4)}


def test_non_units_groups_form_one_decider():
    dist = make_distribution(SPEC, _params())
    assert isinstance(dist, TreeDist) and dist.num_deciders == 1 and dist.batch_size == 2
    actions = {"move": torch.tensor([0, 2]), "aim": torch.zeros(2, 2), "build": torch.tensor([[1, 0], [0, 1]])}
    expected = -math.log(3) - math.log(2 * math.pi) - 2 * math.log(2)
    assert torch.allclose(dist.unit_log_prob(actions), torch.full((2, 1), expected))
    assert torch.allclose(dist.log_prob(actions), torch.full((2,), expected))
    assert dist.unit_valid(actions).tolist() == [[True], [True]]
    entropy = math.log(3) + 1.0 + math.log(2 * math.pi) + 2 * math.log(2)
    assert torch.allclose(dist.unit_entropy(actions), torch.full((2, 1), entropy))
    assert torch.allclose(dist.unit_kl(dist, actions), torch.zeros(2, 1), atol=1e-6)


def test_sample_and_mode_are_trees_in_natural_order():
    dist = make_distribution(SPEC, _params(batch=5))
    sample = dist.sample()
    assert list(sample) == ["move", "aim", "build"]
    assert sample["move"].shape == (5,) and sample["move"].dtype == torch.int64
    assert sample["aim"].shape == (5, 2) and sample["aim"].dtype == torch.float32
    assert sample["build"].shape == (5, 2) and sample["build"].dtype == torch.int64
    assert list(dist.mode()) == ["move", "aim", "build"]


def test_single_group_spaces_use_the_bare_value():
    dist = make_distribution(ActionSpec.from_space(Discrete(4)), torch.zeros(3, 4))
    assert dist.sample().shape == (3,)
    assert torch.allclose(dist.log_prob(torch.tensor([0, 1, 2])), torch.full((3,), -math.log(4)))


def test_apply_mask_tree_reaches_each_group():
    dist = make_distribution(SPEC, _params()).apply_mask(
        {"move": torch.tensor([[True, False, False], [True, True, True]]), "build": torch.ones(2, 4, dtype=bool)})
    assert dist.mode()["move"].tolist() == [0, 0]
    actions = {"move": torch.tensor([0, 1]), "aim": torch.zeros(2, 2), "build": torch.zeros(2, 2, dtype=torch.long)}
    lp = dist.unit_log_prob(actions)[:, 0] + math.log(2 * math.pi) + 2 * math.log(2)
    assert torch.allclose(lp, torch.tensor([0.0, -math.log(3)]), atol=1e-6)
    plain = make_distribution(SPEC, _params())
    assert plain.apply_mask(None) is plain


def test_cat_matches_separate_evaluation():
    a, b = make_distribution(SPEC, _params(1)), make_distribution(SPEC, _params(2))
    joined = TreeDist.cat([a, b])
    assert joined.batch_size == 3
    actions = {"move": torch.tensor([0, 1, 2]), "aim": torch.zeros(3, 2), "build": torch.zeros(3, 2, dtype=torch.long)}
    first = {"move": actions["move"][:1], "aim": actions["aim"][:1], "build": actions["build"][:1]}
    assert torch.allclose(joined.log_prob(actions)[:1], a.log_prob(first))


def test_parameter_errors():
    with pytest.raises(ValueError, match="needs 3 logits, got 4"):
        make_distribution(SPEC, {**_params(), "move": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="no parameters for action group build"):
        make_distribution(SPEC, {"move": torch.zeros(2, 3), "aim": _params()["aim"]})
    with pytest.raises(ValueError, match="parts for groups"):
        TreeDist(SPEC, {("move",): CategoricalDist(torch.zeros(2, 3))})
    with pytest.raises(ValueError, match="disagree on the batch size"):
        make_distribution(SPEC, {**_params(), "move": torch.zeros(3, 3)})


def test_kl_needs_the_same_action_space():
    dist = make_distribution(SPEC, _params())
    other = make_distribution(ActionSpec.from_space(Discrete(3)), torch.zeros(2, 3))
    with pytest.raises(TypeError):
        dist.unit_kl(other, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_leaf_distributions.py tests/unit/test_tree_distribution.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.networks'`.

- [ ] **Step 3: Implement the protocol and the leaf distributions**

Create the empty package files:

```bash
touch src/colosseum/sp2/networks/__init__.py
```

Create `src/colosseum/sp2/networks/dist/base.py`:

```python
"""The decider-aware distribution protocol (SP2 spec block 2, "Протокол распределения").

An action splits into K *deciders*: every unit of every ``Units`` group is one decider, and
all non-units parts together are one more (decider 0, if any). Per-decider methods return
``[B, K]``; invalid deciders (absent units, units without a valid component) give exactly 0,
selected with ``torch.where`` (never by multiplying with a mask: ``0 * NaN = NaN``).

The recorded action is passed to every per-decider method, because ``only_if`` makes the
validity of a component depend on the chosen value of its parent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from torch import Tensor

from colosseum.sp2.core.tree import Tree


class Distribution(ABC):
    """Action distribution over a batch ``[B]`` with ``num_deciders`` deciders."""

    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abstractmethod
    def num_deciders(self) -> int: ...

    @abstractmethod
    def sample(self) -> Tree:
        """A sampled action tree (torch, batch first)."""

    @abstractmethod
    def mode(self) -> Tree:
        """The most likely action tree (argmax / mean)."""

    def log_prob(self, actions: Tree) -> Tensor:
        """Joint log-probability ``[B]``: the sum over valid deciders of :meth:`unit_log_prob`."""
        return self.unit_log_prob(actions).sum(dim=-1)

    @abstractmethod
    def unit_log_prob(self, actions: Tree) -> Tensor:
        """``[B, K]`` log-probability per decider; 0 where invalid."""

    @abstractmethod
    def unit_entropy(self, actions: Tree) -> Tensor:
        """``[B, K]`` entropy per decider (gated by ``only_if`` on ``actions``); 0 where invalid."""

    @abstractmethod
    def unit_valid(self, actions: Tree) -> Tensor:
        """``[B, K]`` bool: the decider counts for ``actions``."""

    @abstractmethod
    def unit_kl(self, other: Distribution, actions: Tree) -> Tensor:
        """``[B, K]`` KL(self || other) per decider; 0 where invalid."""

    @abstractmethod
    def apply_mask(self, mask: Tree | None) -> Distribution:
        """A new distribution with ``mask`` combined into the current one (None: unchanged)."""

    @classmethod
    @abstractmethod
    def cat(cls, dists: Sequence[Distribution]) -> Distribution:
        """Concatenate along the batch dimension."""
```

Create `src/colosseum/sp2/networks/dist/leaf.py`:

```python
"""Leaf distributions: categorical, multi-categorical, diagonal Gaussian. Alone, each is ONE decider."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from colosseum.sp2.networks.dist.base import Distribution


def masked_log_softmax(logits: Tensor, mask: Tensor | None) -> Tensor:
    """log_softmax over the last dim with illegal entries at ``finfo.min`` (an all-illegal row stays finite)."""
    if mask is not None:
        logits = torch.where(mask, logits, torch.finfo(logits.dtype).min)
    return torch.log_softmax(logits, dim=-1)


def categorical_log_prob(log_p: Tensor, actions: Tensor) -> Tensor:
    """``log_p[..., n]`` at integer ``actions[...]``."""
    return log_p.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


def categorical_entropy(log_p: Tensor, mask: Tensor | None) -> Tensor:
    """Entropy over the last dim; illegal entries contribute exactly 0 to the value and the gradient."""
    if mask is not None:
        log_p = torch.where(mask, log_p, torch.zeros_like(log_p))  # exp(0) * 0 = 0, no gradient to illegal logits
    return -(log_p.exp() * log_p).sum(dim=-1)


def categorical_kl(p_logits: Tensor, p_mask: Tensor | None, q_logits: Tensor, q_mask: Tensor | None) -> Tensor:
    """KL(p || q) over the last dim on the intersection of both masks (both renormalized on it), float32."""
    legal = p_mask
    if q_mask is not None:
        legal = q_mask if legal is None else legal & q_mask
    log_p = masked_log_softmax(p_logits.float(), legal)
    log_q = masked_log_softmax(q_logits.float(), legal)
    terms = log_p.exp() * (log_p - log_q)
    if legal is not None:
        terms = torch.where(legal, terms, torch.zeros_like(terms))
    return terms.sum(dim=-1)


def sample_categorical(log_p: Tensor) -> Tensor:
    return torch.distributions.Categorical(logits=log_p, validate_args=False).sample()


def gaussian_log_prob(mean: Tensor, log_std: Tensor, x: Tensor) -> Tensor:
    """Sum over the last dim of the Normal log-density."""
    var = torch.exp(2.0 * log_std)
    return (-((x - mean) ** 2) / (2.0 * var) - log_std - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)


def gaussian_entropy(log_std: Tensor) -> Tensor:
    return (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1)


def gaussian_kl(mean_p: Tensor, log_std_p: Tensor, mean_q: Tensor, log_std_q: Tensor) -> Tensor:
    var_p, var_q = torch.exp(2.0 * log_std_p), torch.exp(2.0 * log_std_q)
    return (log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q) - 0.5).sum(dim=-1)


def _cat_masks(masks: Sequence[Tensor | None], shapes: Sequence[torch.Size], device: torch.device) -> Tensor | None:
    if all(m is None for m in masks):
        return None
    return torch.cat([m if m is not None else torch.ones(s, dtype=torch.bool, device=device)
                      for m, s in zip(masks, shapes)], dim=0)


class CategoricalDist(Distribution):
    """``logits [B, n]``, optional ``mask [B, n]`` (True = legal). Actions ``int64 [B]``."""

    def __init__(self, logits: Tensor, mask: Tensor | None = None) -> None:
        if logits.dim() != 2:
            raise ValueError(f"CategoricalDist: logits must be [B, n], got shape {tuple(logits.shape)}")
        self._logits = logits
        self._mask = None if mask is None else mask.to(device=logits.device, dtype=torch.bool)
        if self._mask is not None and self._mask.shape != logits.shape:
            raise ValueError(f"CategoricalDist: mask shape {tuple(self._mask.shape)} != logits shape "
                             f"{tuple(logits.shape)}")
        self._log_p = masked_log_softmax(logits, self._mask)

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mask(self) -> Tensor | None:
        return self._mask

    @property
    def num_categories(self) -> int:
        return int(self._logits.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self._logits.shape[0])

    @property
    def num_deciders(self) -> int:
        return 1

    def sample(self) -> Tensor:
        return sample_categorical(self._log_p)

    def mode(self) -> Tensor:
        return self._log_p.argmax(dim=-1)

    def unit_log_prob(self, actions: Tensor) -> Tensor:
        return categorical_log_prob(self._log_p, actions).unsqueeze(-1)

    def unit_entropy(self, actions: Tensor) -> Tensor:
        return categorical_entropy(self._log_p, self._mask).unsqueeze(-1)

    def unit_valid(self, actions: Tensor) -> Tensor:
        return torch.ones(self.batch_size, 1, dtype=torch.bool, device=self._logits.device)

    def unit_kl(self, other: Distribution, actions: Tensor) -> Tensor:
        if not isinstance(other, CategoricalDist):
            raise TypeError(f"KL between CategoricalDist and {type(other).__name__} is not defined")
        return categorical_kl(self._logits, self._mask, other._logits, other._mask).unsqueeze(-1)

    def apply_mask(self, mask: Tensor | None) -> CategoricalDist:
        if mask is None:
            return self
        mask = mask.to(device=self._logits.device, dtype=torch.bool)
        return CategoricalDist(self._logits, mask if self._mask is None else mask & self._mask)

    @classmethod
    def cat(cls, dists: Sequence[CategoricalDist]) -> CategoricalDist:
        logits = torch.cat([d._logits for d in dists], dim=0)
        mask = _cat_masks([d._mask for d in dists], [d._logits.shape for d in dists], logits.device)
        return CategoricalDist(logits, mask)


class MultiCategoricalDist(Distribution):
    """Independent categoricals in one decider: ``logits [B, sum(nvec)]``, ``mask [B, sum(nvec)]``.
    Actions ``int64 [B, C]``."""

    def __init__(self, logits: Tensor, nvec: Sequence[int], mask: Tensor | None = None) -> None:
        self.nvec = tuple(int(n) for n in nvec)
        if logits.dim() != 2 or logits.shape[-1] != sum(self.nvec):
            raise ValueError(f"MultiCategoricalDist: logits must be [B, {sum(self.nvec)}] for nvec {self.nvec}, "
                             f"got shape {tuple(logits.shape)}")
        self._logits = logits
        self._mask = None if mask is None else mask.to(device=logits.device, dtype=torch.bool)
        if self._mask is not None and self._mask.shape != logits.shape:
            raise ValueError(f"MultiCategoricalDist: mask shape {tuple(self._mask.shape)} != logits shape "
                             f"{tuple(logits.shape)}")
        self._parts: list[tuple[Tensor, Tensor | None]] = []
        offset = 0
        for n in self.nvec:
            m = None if self._mask is None else self._mask[:, offset:offset + n]
            self._parts.append((masked_log_softmax(logits[:, offset:offset + n], m), m))
            offset += n

    @property
    def logits(self) -> Tensor:
        return self._logits

    @property
    def mask(self) -> Tensor | None:
        return self._mask

    @property
    def batch_size(self) -> int:
        return int(self._logits.shape[0])

    @property
    def num_deciders(self) -> int:
        return 1

    def sample(self) -> Tensor:
        return torch.stack([sample_categorical(log_p) for log_p, _ in self._parts], dim=-1)

    def mode(self) -> Tensor:
        return torch.stack([log_p.argmax(dim=-1) for log_p, _ in self._parts], dim=-1)

    def unit_log_prob(self, actions: Tensor) -> Tensor:
        return sum(categorical_log_prob(log_p, actions[:, i]) for i, (log_p, _) in enumerate(self._parts)).unsqueeze(-1)

    def unit_entropy(self, actions: Tensor) -> Tensor:
        return sum(categorical_entropy(log_p, m) for log_p, m in self._parts).unsqueeze(-1)

    def unit_valid(self, actions: Tensor) -> Tensor:
        return torch.ones(self.batch_size, 1, dtype=torch.bool, device=self._logits.device)

    def unit_kl(self, other: Distribution, actions: Tensor) -> Tensor:
        if not isinstance(other, MultiCategoricalDist) or other.nvec != self.nvec:
            raise TypeError(f"KL between MultiCategoricalDist{self.nvec} and {type(other).__name__} is not defined")
        total, offset = None, 0
        for n in self.nvec:
            sl = slice(offset, offset + n)
            kl = categorical_kl(self._logits[:, sl], None if self._mask is None else self._mask[:, sl],
                                other._logits[:, sl], None if other._mask is None else other._mask[:, sl])
            total = kl if total is None else total + kl
            offset += n
        return total.unsqueeze(-1)

    def apply_mask(self, mask: Tensor | None) -> MultiCategoricalDist:
        if mask is None:
            return self
        mask = mask.to(device=self._logits.device, dtype=torch.bool)
        return MultiCategoricalDist(self._logits, self.nvec, mask if self._mask is None else mask & self._mask)

    @classmethod
    def cat(cls, dists: Sequence[MultiCategoricalDist]) -> MultiCategoricalDist:
        logits = torch.cat([d._logits for d in dists], dim=0)
        mask = _cat_masks([d._mask for d in dists], [d._logits.shape for d in dists], logits.device)
        return MultiCategoricalDist(logits, dists[0].nvec, mask)


class DiagGaussianDist(Distribution):
    """``mean [B, d]``, ``log_std [B, d]`` or ``[d]``. Actions ``float [B, d]``; no mask."""

    def __init__(self, mean: Tensor, log_std: Tensor) -> None:
        if mean.dim() != 2:
            raise ValueError(f"DiagGaussianDist: mean must be [B, d], got shape {tuple(mean.shape)}")
        self._mean = mean
        self._log_std = torch.broadcast_to(log_std, mean.shape)

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def log_std(self) -> Tensor:
        return self._log_std

    @property
    def action_dim(self) -> int:
        return int(self._mean.shape[-1])

    @property
    def batch_size(self) -> int:
        return int(self._mean.shape[0])

    @property
    def num_deciders(self) -> int:
        return 1

    def sample(self) -> Tensor:
        with torch.no_grad():
            return self._mean + torch.exp(self._log_std) * torch.randn_like(self._mean)

    def mode(self) -> Tensor:
        return self._mean

    def unit_log_prob(self, actions: Tensor) -> Tensor:
        return gaussian_log_prob(self._mean, self._log_std, actions.to(self._mean.dtype)).unsqueeze(-1)

    def unit_entropy(self, actions: Tensor) -> Tensor:
        return gaussian_entropy(self._log_std).unsqueeze(-1)

    def unit_valid(self, actions: Tensor) -> Tensor:
        return torch.ones(self.batch_size, 1, dtype=torch.bool, device=self._mean.device)

    def unit_kl(self, other: Distribution, actions: Tensor) -> Tensor:
        if not isinstance(other, DiagGaussianDist):
            raise TypeError(f"KL between DiagGaussianDist and {type(other).__name__} is not defined")
        return gaussian_kl(self._mean, self._log_std, other._mean, other._log_std).unsqueeze(-1)

    def apply_mask(self, mask: Tensor | None) -> DiagGaussianDist:
        if mask is not None:
            raise ValueError("DiagGaussianDist (a Box action) takes no mask")
        return self

    @classmethod
    def cat(cls, dists: Sequence[DiagGaussianDist]) -> DiagGaussianDist:
        return DiagGaussianDist(torch.cat([d._mean for d in dists], dim=0),
                                torch.cat([d._log_std for d in dists], dim=0))
```

- [ ] **Step 4: Implement `TreeDist` and `make_distribution`**

Create `src/colosseum/sp2/networks/dist/tree.py`:

```python
"""``TreeDist``: one distribution per action group of an ``ActionSpec``, and ``make_distribution``.

Deciders: decider 0 is all non-units groups together (their per-decider values are summed),
then each units group in spec order contributes its ``max_units`` deciders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from colosseum.sp2.core.specs import ActionGroup, ActionSpec
from colosseum.sp2.core.tree import Tree, tree_get
from colosseum.sp2.networks.dist.base import Distribution
from colosseum.sp2.networks.dist.leaf import CategoricalDist, DiagGaussianDist, MultiCategoricalDist


def _fmt(path: tuple[str, ...]) -> str:
    return "/".join(path) if path else "<root>"


def _check_part(group: ActionGroup, part: Distribution) -> None:
    where = f"action group {_fmt(group.path)}"
    if not isinstance(part, Distribution):
        raise TypeError(f"{where}: expected a Distribution, got {type(part).__name__}")
    if group.kind == "units":
        if part.num_deciders != group.units.max_units:
            raise ValueError(f"{where}: Units({group.units.max_units}) needs a distribution with "
                             f"{group.units.max_units} deciders, got {part.num_deciders}")
        return
    if part.num_deciders != 1:
        raise ValueError(f"{where}: a {group.kind} group needs a one-decider distribution, got {part.num_deciders}")
    if group.kind == "discrete" and isinstance(part, CategoricalDist) and part.num_categories != group.nvec[0]:
        raise ValueError(f"{where}: Discrete({group.nvec[0]}) needs {group.nvec[0]} logits, got {part.num_categories}")
    if group.kind == "multi_discrete" and isinstance(part, MultiCategoricalDist) and part.nvec != group.nvec:
        raise ValueError(f"{where}: MultiDiscrete{list(group.nvec)} got a MultiCategoricalDist with nvec "
                         f"{list(part.nvec)}")
    if group.kind == "box" and isinstance(part, DiagGaussianDist) and part.action_dim != group.box_dim:
        raise ValueError(f"{where}: Box({group.box_dim},) got a DiagGaussianDist of dimension {part.action_dim}")
    expected = {"discrete": CategoricalDist, "multi_discrete": MultiCategoricalDist, "box": DiagGaussianDist}
    builtin = (CategoricalDist, MultiCategoricalDist, DiagGaussianDist)
    if isinstance(part, builtin) and not isinstance(part, expected[group.kind]):
        raise ValueError(f"{where}: a {group.kind} group needs a {expected[group.kind].__name__}, "
                         f"got {type(part).__name__}")


class TreeDist(Distribution):
    """Composition of per-group distributions along an ``ActionSpec``."""

    def __init__(self, spec: ActionSpec, parts: Mapping[tuple[str, ...], Distribution]) -> None:
        paths = [g.path for g in spec.groups]
        if set(parts) != set(paths):
            raise ValueError(f"TreeDist: parts for groups {sorted(map(_fmt, parts))}, the action space has "
                             f"{[_fmt(p) for p in paths]}")
        self.spec = spec
        self.parts: dict[tuple[str, ...], Distribution] = {p: parts[p] for p in paths}
        for group in spec.groups:
            _check_part(group, self.parts[group.path])
        sizes = {part.batch_size for part in self.parts.values()}
        if len(sizes) != 1:
            raise ValueError(f"TreeDist: parts disagree on the batch size: {sorted(sizes)}")
        self._flat = [g for g in spec.groups if g.kind != "units"]
        self._units = [g for g in spec.groups if g.kind == "units"]

    @property
    def batch_size(self) -> int:
        return next(iter(self.parts.values())).batch_size

    @property
    def num_deciders(self) -> int:
        return self.spec.num_deciders

    def _action(self, actions: Tree, group: ActionGroup) -> Any:
        return actions if not self.spec.is_dict else tree_get(actions, group.path)

    def _assemble(self, values: dict[tuple[str, ...], Any]) -> Tree:
        if not self.spec.is_dict:
            return values[self.spec.groups[0].path]
        out: dict = {}
        for path, value in values.items():
            node = out
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value
        return out

    def sample(self) -> Tree:
        return self._assemble({p: part.sample() for p, part in self.parts.items()})

    def mode(self) -> Tree:
        return self._assemble({p: part.mode() for p, part in self.parts.items()})

    def _per_decider(self, fn: Any, actions: Tree) -> Tensor:
        columns: list[Tensor] = []
        if self._flat:
            columns.append(sum(fn(self.parts[g.path], self._action(actions, g)) for g in self._flat))
        columns.extend(fn(self.parts[g.path], self._action(actions, g)) for g in self._units)
        return torch.cat(columns, dim=-1)

    def unit_log_prob(self, actions: Tree) -> Tensor:
        return self._per_decider(lambda part, a: part.unit_log_prob(a), actions)

    def unit_entropy(self, actions: Tree) -> Tensor:
        return self._per_decider(lambda part, a: part.unit_entropy(a), actions)

    def unit_valid(self, actions: Tree) -> Tensor:
        columns: list[Tensor] = []
        if self._flat:
            device = self.parts[self._flat[0].path].unit_valid(self._action(actions, self._flat[0])).device
            columns.append(torch.ones(self.batch_size, 1, dtype=torch.bool, device=device))
        columns.extend(self.parts[g.path].unit_valid(self._action(actions, g)) for g in self._units)
        return torch.cat(columns, dim=-1)

    def unit_kl(self, other: Distribution, actions: Tree) -> Tensor:
        if not isinstance(other, TreeDist) or other.spec != self.spec:
            raise TypeError(f"KL between TreeDist and {type(other).__name__} over a different action space")
        columns: list[Tensor] = []
        if self._flat:
            columns.append(sum(self.parts[g.path].unit_kl(other.parts[g.path], self._action(actions, g))
                               for g in self._flat))
        columns.extend(self.parts[g.path].unit_kl(other.parts[g.path], self._action(actions, g)) for g in self._units)
        return torch.cat(columns, dim=-1)

    def apply_mask(self, mask: Tree | None) -> TreeDist:
        if mask is None:
            return self
        parts = {}
        for group in self.spec.groups:
            part = self.parts[group.path]
            group_mask = self.spec.group_mask(mask, group)
            parts[group.path] = part if group_mask is None else part.apply_mask(group_mask)
        return TreeDist(self.spec, parts)

    @classmethod
    def cat(cls, dists: Sequence[TreeDist]) -> TreeDist:
        spec = dists[0].spec
        return TreeDist(spec, {p: type(part).cat([d.parts[p] for d in dists]) for p, part in dists[0].parts.items()})


def _group_params(params: Tree, spec: ActionSpec, group: ActionGroup) -> Any:
    if not spec.is_dict:
        return params
    try:
        return tree_get(params, group.path)
    except KeyError:
        raise ValueError(f"make_distribution: no parameters for action group {_fmt(group.path)}") from None


def make_distribution(spec: ActionSpec, params: Tree) -> TreeDist:
    """Built-in distribution for ``spec`` from a parameter tree that mirrors the action tree.

    discrete -> logits ``[B, n]``; multi_discrete -> logits ``[B, sum(nvec)]``; box ->
    ``{"mean": [B, d], "log_std": [B, d] | [d]}``; units -> ``UnitsDist`` parameters
    (one entry per component: logits ``[B, U, n]`` or ``{"mean", "log_std"}``).
    """
    parts: dict[tuple[str, ...], Distribution] = {}
    for group in spec.groups:
        p = _group_params(params, spec, group)
        if group.kind == "discrete":
            parts[group.path] = CategoricalDist(p)
        elif group.kind == "multi_discrete":
            parts[group.path] = MultiCategoricalDist(p, group.nvec)
        elif group.kind == "box":
            parts[group.path] = DiagGaussianDist(p["mean"], p["log_std"])
        else:
            raise NotImplementedError("Units action groups need UnitsDist (added in T2.2)")
    return TreeDist(spec, parts)
```

Create `src/colosseum/sp2/networks/dist/__init__.py`:

```python
"""Action distributions over decider trees (SP2 spec block 2)."""

from colosseum.sp2.networks.dist.base import Distribution
from colosseum.sp2.networks.dist.leaf import CategoricalDist, DiagGaussianDist, MultiCategoricalDist
from colosseum.sp2.networks.dist.tree import TreeDist, make_distribution

__all__ = ["CategoricalDist", "DiagGaussianDist", "Distribution", "MultiCategoricalDist", "TreeDist",
           "make_distribution"]
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_leaf_distributions.py tests/unit/test_tree_distribution.py -q`
Expected: `17 passed`.

- [ ] **Step 6: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/sp2/networks/__init__.py \
  src/colosseum/sp2/networks/dist/__init__.py \
  src/colosseum/sp2/networks/dist/base.py \
  src/colosseum/sp2/networks/dist/leaf.py \
  src/colosseum/sp2/networks/dist/tree.py \
  tests/unit/test_leaf_distributions.py \
  tests/unit/test_tree_distribution.py
git commit -m "feat(sp2): decider-aware distribution protocol, leaf distributions and TreeDist"
git push origin sp2-game-model
```

---

### Task T2.2: `UnitsDist`: per-unit deciders, masks and `only_if`

Spec block 2, "`Units`", "Правило пустой строки", "Протокол распределения". `UnitsDist` is one units group: K = `max_units` deciders. A unit is a valid decider iff its `unit` mask is True and at least one component is valid; a discrete component is valid iff its mask row is non-empty and its `only_if` gate holds; a box component iff its gate holds; the gate of `child <- (parent, values)` holds iff the parent component is valid and the GIVEN action's parent value is in `values` (on the worker the sampled action, on the learner the recorded one). Log-prob, entropy and KL of a unit sum its valid components; everything invalid is exactly 0 with no gradient (absent units, empty rows) — `test_dead_units_and_empty_rows_give_no_nan_and_no_gradient` runs the backward pass under a 65536 loss scale. `sample`/`mode` return 0 for absent units and empty rows (the env ignores them).

Exact values pinned by the tests (zero logits, `move: Discrete(3)`, `target: Discrete(2)` counted only when `move == 2`, unit 2 absent, actions `move=[2, 0, 1]`): `unit_log_prob = [-ln3 - ln2, -ln3, 0]`, `unit_entropy = [ln3 + ln2, ln3, 0]`, `unit_valid = [T, T, F]`.

**Files:**
- Create: `src/colosseum/sp2/networks/dist/units.py`
- Modify: `src/colosseum/sp2/networks/dist/tree.py` (`make_distribution` builds `UnitsDist`)
- Modify: `src/colosseum/sp2/networks/dist/__init__.py` (export `UnitsDist`)
- Test: `tests/unit/test_units_distribution.py`

**Interfaces:**
- Consumes: `ActionGroup`, `ActionSpec` (T1.3); `UnitComponent` (T1.2); the leaf helpers of `colosseum.sp2.networks.dist.leaf` and `TreeDist` (T2.1); `tree_to_torch` (T1.1, tests).
- Produces: `colosseum.sp2.networks.dist.UnitsDist(group, params, mask=None)` — the contract (`params`: discrete component -> logits `[B, U, n]`; box component -> `{"mean": [B, U, d], "log_std": [B, U, d] | [d]}`; `mask`: `{"unit": [B, U], "action": [B, U, sum discrete sizes]}`, keys optional). Attributes used by tests and later tasks: `group`, `units`, `U`, `B`, `params`, `unit_mask`, `action_mask`. `make_distribution` now supports every group kind.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_units_distribution.py`:

```python
"""UnitsDist: per-unit deciders, masks, only_if gating by the given action, no NaN (SP2 T2.2)."""
import math

import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_to_torch
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.dist import UnitsDist, make_distribution

LOG2, LOG3 = math.log(2), math.log(3)

# move (3) | target (2); target counts only when move == 2.
MOVE_TARGET = Units(3, Dict([("move", Discrete(3)), ("target", Discrete(2))]), only_if={"target": ("move", {2})})
GROUP = ActionSpec.from_space(MOVE_TARGET).groups[0]


def _dist(mask=None, batch=1):
    params = {"move": torch.zeros(batch, 3, 3), "target": torch.zeros(batch, 3, 2)}
    return UnitsDist(GROUP, params, mask)


def _actions(move, target):
    return {"move": torch.tensor([move]), "target": torch.tensor([target])}


def test_exact_log_prob_entropy_and_validity_with_only_if():
    dist = _dist({"unit": torch.tensor([[True, True, False]])})
    a = _actions([2, 0, 1], [1, 1, 1])
    assert dist.num_deciders == 3 and dist.batch_size == 1
    # unit 0: move=2 -> target counts; unit 1: move=0 -> target gated off; unit 2: absent.
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG3 - LOG2, -LOG3, 0.0]]))
    assert torch.allclose(dist.log_prob(a), torch.tensor([-2 * LOG3 - LOG2]))
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[LOG3 + LOG2, LOG3, 0.0]]))
    assert dist.unit_valid(a).tolist() == [[True, True, False]]
    # The gate follows the given action: with move=2 on unit 1 its target counts too.
    assert torch.allclose(dist.unit_entropy(_actions([2, 2, 1], [0, 0, 0])), torch.tensor([[LOG3 + LOG2] * 2 + [0.0]]))


def test_action_mask_rows_and_empty_rows():
    # unit 0: move row [T, F, T], target row [F, T]; unit 1: empty move row -> move invalid,
    # target gated by an invalid parent -> invalid -> the whole unit is invalid.
    action = torch.tensor([[[True, False, True, False, True],
                            [False, False, False, True, True],
                            [True, True, True, True, True]]])
    dist = _dist({"unit": torch.tensor([[True, True, False]]), "action": action})
    a = _actions([2, 2, 0], [1, 0, 0])
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2 + 0.0, 0.0, 0.0]]))
    assert dist.unit_valid(a).tolist() == [[True, False, False]]
    assert torch.allclose(dist.unit_entropy(a), torch.tensor([[LOG2, 0.0, 0.0]]))


def test_dead_units_and_empty_rows_give_no_nan_and_no_gradient():
    move = torch.randn(2, 3, 3, requires_grad=True)
    target = torch.randn(2, 3, 2, requires_grad=True)
    action = torch.zeros(2, 3, 5, dtype=torch.bool)
    action[:, 0] = True
    dist = UnitsDist(GROUP, {"move": move, "target": target},
                     {"unit": torch.tensor([[True, False, True]] * 2), "action": action})
    a = {"move": torch.tensor([[2, 1, 0]] * 2), "target": torch.tensor([[1, 0, 1]] * 2)}
    loss = (dist.log_prob(a).sum() + dist.unit_entropy(a).sum()) * 65536.0
    other = UnitsDist(GROUP, {"move": torch.zeros(2, 3, 3), "target": torch.zeros(2, 3, 2)})
    loss = loss + dist.unit_kl(other, a).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(move.grad).all() and torch.isfinite(target.grad).all()
    assert torch.all(move.grad[:, 1:] == 0) and torch.all(target.grad[:, 1:] == 0)
    assert dist.unit_valid(a).tolist() == [[True, False, False]] * 2


def test_sample_and_mode_zero_absent_units_and_respect_masks():
    action = torch.ones(1, 3, 5, dtype=torch.bool)
    action[0, 0, :3] = torch.tensor([False, True, False])
    dist = _dist({"unit": torch.tensor([[True, True, False]]), "action": action})
    torch.manual_seed(0)
    for _ in range(20):
        s = dist.sample()
        assert s["move"].shape == (1, 3) and s["move"].dtype == torch.int64
        assert s["move"][0, 0].item() == 1 and s["move"][0, 2].item() == 0 and s["target"][0, 2].item() == 0
    assert dist.mode()["move"].tolist() == [[1, 0, 0]]


def test_kl_is_gated_like_the_entropy():
    p = _dist()
    q = UnitsDist(GROUP, {"move": torch.zeros(1, 3, 3), "target": torch.log(torch.tensor([[[0.25, 0.75]] * 3]))})
    a = _actions([2, 0, 2], [0, 0, 0])
    kl_target = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert torch.allclose(p.unit_kl(q, a), torch.tensor([[kl_target, 0.0, kl_target]]), atol=1e-6)


def test_multi_discrete_and_box_components():
    md = ActionSpec.from_space(Units(2, MultiDiscrete([2, 4]), only_if={"1": ("0", {1})})).groups[0]
    dist = UnitsDist(md, {"0": torch.zeros(1, 2, 2), "1": torch.zeros(1, 2, 4)})
    a = torch.tensor([[[1, 3], [0, 3]]])
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2 - math.log(4), -LOG2]]))
    assert dist.sample().shape == (1, 2, 2)
    box = ActionSpec.from_space(Units(2, Dict([("kind", Discrete(2)), ("thrust", Box(-1.0, 1.0, (2,)))]),
                                      only_if={"thrust": ("kind", {1})})).groups[0]
    dist = UnitsDist(box, {"kind": torch.zeros(1, 2, 2),
                           "thrust": {"mean": torch.zeros(1, 2, 2), "log_std": torch.zeros(2)}})
    a = {"kind": torch.tensor([[1, 0]]), "thrust": torch.zeros(1, 2, 2)}
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-LOG2 - math.log(2 * math.pi), -LOG2]]))
    s = dist.sample()
    assert s["thrust"].shape == (1, 2, 2) and s["thrust"].dtype == torch.float32


def test_box_only_units_are_valid_when_present():
    group = ActionSpec.from_space(Units(2, Box(-1.0, 1.0, (1,)))).groups[0]
    dist = UnitsDist(group, {"0": {"mean": torch.zeros(1, 2, 1), "log_std": torch.zeros(1)}},
                     {"unit": torch.tensor([[True, False]])})
    a = torch.zeros(1, 2, 1)
    assert dist.unit_valid(a).tolist() == [[True, False]]
    assert torch.allclose(dist.unit_log_prob(a), torch.tensor([[-0.5 * math.log(2 * math.pi), 0.0]]))


def test_apply_mask_and_cat():
    dist = _dist(batch=2).apply_mask({"unit": torch.tensor([[True, False, True], [False, False, True]])})
    dist = dist.apply_mask({"unit": torch.tensor([[True, True, False], [True, True, True]])})
    assert dist.unit_mask.tolist() == [[True, False, False], [False, False, True]]
    with_action = dist.apply_mask({"action": torch.ones(2, 3, 5, dtype=torch.bool)})
    joined = UnitsDist.cat([with_action, _dist()])
    assert joined.batch_size == 3 and joined.action_mask.shape == (3, 3, 5) and joined.action_mask.all()
    assert joined.unit_mask[2].tolist() == [True, True, True]


def test_parameter_errors():
    with pytest.raises(ValueError, match="params for components"):
        UnitsDist(GROUP, {"move": torch.zeros(1, 3, 3)})
    with pytest.raises(ValueError, match=r"needs logits \[B, 3, 2\]"):
        UnitsDist(GROUP, {"move": torch.zeros(1, 3, 3), "target": torch.zeros(1, 3, 3)})
    with pytest.raises(ValueError, match=r"action mask must be \[B, 3, 5\]"):
        _dist({"action": torch.ones(1, 3, 4, dtype=torch.bool)})


def test_tree_dist_lays_out_deciders():
    spec = ActionSpec.from_space(Dict([("base", Discrete(3)), ("workers", MOVE_TARGET),
                                       ("scouts", Units(2, Discrete(2)))]))
    assert spec.num_deciders == 1 + 3 + 2
    params = {"base": torch.zeros(1, 3), "workers": {"move": torch.zeros(1, 3, 3), "target": torch.zeros(1, 3, 2)},
              "scouts": {"0": torch.zeros(1, 2, 2)}}
    mask = spec.full_mask((1,))
    mask["workers"]["unit"][0, 2] = False
    mask["scouts"]["unit"][0, 0] = False
    dist = make_distribution(spec, params).apply_mask(tree_to_torch(mask))
    a = {"base": torch.tensor([1]), "workers": _actions([2, 0, 0], [0, 0, 0]), "scouts": torch.tensor([[1, 0]])}
    expected = torch.tensor([[-LOG3, -LOG3 - LOG2, -LOG3, 0.0, 0.0, -LOG2]])
    assert torch.allclose(dist.unit_log_prob(a), expected)
    assert torch.allclose(dist.log_prob(a), expected.sum(-1))
    assert dist.unit_valid(a).tolist() == [[True, True, True, False, False, True]]
    s = dist.sample()
    assert list(s) == ["base", "workers", "scouts"] and s["scouts"].shape == (1, 2)


def test_k1_units_is_one_decider():
    spec = ActionSpec.from_space(Units(1, Dict([("kind", Discrete(2)), ("target", Discrete(4))]),
                                       only_if={"target": ("kind", {1})}))
    dist = make_distribution(spec, {"kind": torch.zeros(1, 1, 2), "target": torch.zeros(1, 1, 4)})
    assert dist.num_deciders == 1
    a = {"kind": torch.tensor([[1]]), "target": torch.tensor([[3]])}
    assert torch.allclose(dist.log_prob(a), torch.tensor([-LOG2 - math.log(4)]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_units_distribution.py -q`
Expected: collection error `ImportError: cannot import name 'UnitsDist' from 'colosseum.sp2.networks.dist'`.

- [ ] **Step 3: Implement `UnitsDist`**

Create `src/colosseum/sp2/networks/dist/units.py`:

```python
"""``UnitsDist``: per-unit components of one ``Units`` group, with masks and ``only_if`` (SP2 block 2).

K = ``max_units`` deciders (one per unit). A unit is a valid decider iff its ``unit`` mask is
True and at least one of its components is valid:
- a discrete component is valid iff its mask row is non-empty and its ``only_if`` gate holds;
- a box component is valid iff its ``only_if`` gate holds;
- the gate of ``child <- (parent, values)`` holds iff the parent component is valid and the
  given action's parent value is in ``values``.
Log-prob, entropy and KL sum the valid components of a unit; invalid positions are 0 via
``torch.where`` (no NaN, no gradient).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from colosseum.sp2.core.specs import ActionGroup
from colosseum.sp2.envs.spaces import UnitComponent
from colosseum.sp2.networks.dist.base import Distribution
from colosseum.sp2.networks.dist.leaf import (
    categorical_entropy,
    categorical_kl,
    categorical_log_prob,
    gaussian_entropy,
    gaussian_kl,
    gaussian_log_prob,
    masked_log_softmax,
    sample_categorical,
)


class UnitsDist(Distribution):
    """``params``: discrete component -> logits ``[B, U, n]``; box component -> ``{"mean": [B, U, d],
    "log_std": [B, U, d] | [d]}``. ``mask``: ``{"unit": [B, U], "action": [B, U, sum discrete sizes]}``
    (either key optional). Actions: the env format of the group with a leading ``B``."""

    def __init__(self, group: ActionGroup, params: Mapping[str, Any], mask: Mapping[str, Tensor] | None = None) -> None:
        if group.kind != "units":
            raise ValueError(f"UnitsDist needs a units action group, got {group.kind!r}")
        self.group = group
        self.units = group.units
        self.U = self.units.max_units
        names = [c.name for c in self.units.components]
        if set(params) != set(names):
            raise ValueError(f"UnitsDist: params for components {sorted(params)}, the units have {names}")
        self.params: dict[str, Any] = {}
        first = params[names[0]]
        ref = first if isinstance(first, Tensor) else first["mean"]
        self.B = int(ref.shape[0])
        device = ref.device
        for c in self.units.components:
            p = params[c.name]
            if c.kind == "discrete":
                if not isinstance(p, Tensor) or tuple(p.shape) != (self.B, self.U, c.size):
                    got = tuple(p.shape) if isinstance(p, Tensor) else type(p).__name__
                    raise ValueError(f"UnitsDist: component {c.name!r} needs logits [B, {self.U}, {c.size}], got {got}")
                self.params[c.name] = p
            else:
                if not isinstance(p, Mapping) or set(p) != {"mean", "log_std"}:
                    raise ValueError(f"UnitsDist: box component {c.name!r} needs {{'mean', 'log_std'}}")
                if tuple(p["mean"].shape) != (self.B, self.U, c.size):
                    raise ValueError(f"UnitsDist: component {c.name!r} needs mean [B, {self.U}, {c.size}], "
                                     f"got {tuple(p['mean'].shape)}")
                self.params[c.name] = {"mean": p["mean"], "log_std": torch.broadcast_to(p["log_std"], p["mean"].shape)}
        mask = mask or {}
        unit = mask.get("unit")
        action = mask.get("action")
        self.unit_mask = (torch.ones(self.B, self.U, dtype=torch.bool, device=device) if unit is None
                          else unit.to(device=device, dtype=torch.bool))
        self.action_mask = None if action is None else action.to(device=device, dtype=torch.bool)
        if tuple(self.unit_mask.shape) != (self.B, self.U):
            raise ValueError(f"UnitsDist: unit mask must be [B, {self.U}], got {tuple(self.unit_mask.shape)}")
        if self.action_mask is not None and tuple(self.action_mask.shape) != (self.B, self.U, group.mask_size):
            raise ValueError(f"UnitsDist: action mask must be [B, {self.U}, {group.mask_size}], "
                             f"got {tuple(self.action_mask.shape)}")
        self._row: dict[str, Tensor | None] = {}   # component mask rows [B, U, n]
        self._log_p: dict[str, Tensor] = {}
        offset = 0
        for c in self.units.components:
            if c.kind == "discrete":
                row = None if self.action_mask is None else self.action_mask[..., offset:offset + c.size]
                self._row[c.name] = row
                self._log_p[c.name] = masked_log_softmax(self.params[c.name], row)
                offset += c.size

    # ---- layout --------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.B

    @property
    def num_deciders(self) -> int:
        return self.U

    def _split(self, actions: Any) -> dict[str, Tensor]:
        kind = self.units.per_unit_kind
        if kind == "dict":
            return {c.name: actions[c.name] for c in self.units.components}
        if kind == "multi_discrete":
            return {c.name: actions[..., i] for i, c in enumerate(self.units.components)}
        return {"0": actions}

    def _join(self, values: dict[str, Tensor]) -> Any:
        kind = self.units.per_unit_kind
        if kind == "dict":
            return values
        if kind == "multi_discrete":
            return torch.stack([values[c.name] for c in self.units.components], dim=-1)
        return values["0"]

    def _row_ok(self, c: UnitComponent) -> Tensor:
        row = self._row.get(c.name)
        if row is None:
            return self.unit_mask
        return self.unit_mask & row.any(dim=-1)

    def _component_valid(self, actions: Any) -> dict[str, Tensor]:
        """``[B, U]`` validity per component for the given actions (``only_if`` in dependency order)."""
        a = self._split(actions)
        valid: dict[str, Tensor] = {}
        pending = list(self.units.components)
        while pending:
            for c in list(pending):
                rule = self.units.only_if.get(c.name)
                if rule is not None and rule[0] not in valid:
                    continue  # parent first
                ok = self._row_ok(c) if c.kind == "discrete" else self.unit_mask
                if rule is not None:
                    parent, values = rule
                    n = next(p.size for p in self.units.components if p.name == parent)
                    table = torch.zeros(n, dtype=torch.bool, device=ok.device)
                    table[sorted(values)] = True
                    ok = ok & valid[parent] & table[a[parent].long().clamp(0, n - 1)]
                valid[c.name] = ok
                pending.remove(c)
        return valid

    # ---- protocol ----------------------------------------------------------------------

    def _draw(self, greedy: bool) -> Any:
        values: dict[str, Tensor] = {}
        for c in self.units.components:
            if c.kind == "discrete":
                log_p = self._log_p[c.name]
                x = log_p.argmax(dim=-1) if greedy else sample_categorical(log_p)
                values[c.name] = torch.where(self._row_ok(c), x, torch.zeros_like(x))
            else:
                p = self.params[c.name]
                x = p["mean"] if greedy else p["mean"] + torch.exp(p["log_std"]) * torch.randn_like(p["mean"])
                values[c.name] = torch.where(self.unit_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return self._join(values)

    def sample(self) -> Any:
        with torch.no_grad():
            return self._draw(greedy=False)

    def mode(self) -> Any:
        return self._draw(greedy=True)

    def _sum_valid(self, terms: dict[str, Tensor], valid: dict[str, Tensor]) -> Tensor:
        total = torch.zeros(self.B, self.U, dtype=next(iter(terms.values())).dtype,
                            device=self.unit_mask.device)
        for name, value in terms.items():
            total = total + torch.where(valid[name], value, torch.zeros_like(value))
        return total

    def unit_log_prob(self, actions: Any) -> Tensor:
        a = self._split(actions)
        terms = {}
        for c in self.units.components:
            if c.kind == "discrete":
                terms[c.name] = categorical_log_prob(self._log_p[c.name], a[c.name].long().clamp(0, c.size - 1))
            else:
                p = self.params[c.name]
                terms[c.name] = gaussian_log_prob(p["mean"], p["log_std"], a[c.name].to(p["mean"].dtype))
        return self._sum_valid(terms, self._component_valid(actions))

    def unit_entropy(self, actions: Any) -> Tensor:
        terms = {}
        for c in self.units.components:
            if c.kind == "discrete":
                terms[c.name] = categorical_entropy(self._log_p[c.name], self._row[c.name])
            else:
                terms[c.name] = gaussian_entropy(self.params[c.name]["log_std"])
        return self._sum_valid(terms, self._component_valid(actions))

    def unit_valid(self, actions: Any) -> Tensor:
        valid = self._component_valid(actions)
        any_valid = torch.zeros_like(self.unit_mask)
        for v in valid.values():
            any_valid = any_valid | v
        return self.unit_mask & any_valid

    def unit_kl(self, other: Distribution, actions: Any) -> Tensor:
        if not isinstance(other, UnitsDist) or other.group != self.group:
            raise TypeError(f"KL between UnitsDist and {type(other).__name__} over a different group")
        terms = {}
        for c in self.units.components:
            if c.kind == "discrete":
                terms[c.name] = categorical_kl(self.params[c.name], self._row[c.name],
                                               other.params[c.name], other._row[c.name])
            else:
                p, q = self.params[c.name], other.params[c.name]
                terms[c.name] = gaussian_kl(p["mean"].float(), p["log_std"].float(),
                                            q["mean"].float(), q["log_std"].float())
        return self._sum_valid(terms, self._component_valid(actions))

    def apply_mask(self, mask: Mapping[str, Tensor] | None) -> UnitsDist:
        if mask is None:
            return self
        unit = mask.get("unit")
        action = mask.get("action")
        new_unit = self.unit_mask if unit is None else self.unit_mask & unit.to(self.unit_mask.device, torch.bool)
        if action is None:
            new_action = self.action_mask
        else:
            action = action.to(self.unit_mask.device, torch.bool)
            new_action = action if self.action_mask is None else self.action_mask & action
        combined = {"unit": new_unit}
        if new_action is not None:
            combined["action"] = new_action
        return UnitsDist(self.group, self.params, combined)

    @classmethod
    def cat(cls, dists: Sequence[UnitsDist]) -> UnitsDist:
        first = dists[0]
        params: dict[str, Any] = {}
        for c in first.units.components:
            if c.kind == "discrete":
                params[c.name] = torch.cat([d.params[c.name] for d in dists], dim=0)
            else:
                params[c.name] = {k: torch.cat([d.params[c.name][k] for d in dists], dim=0)
                                  for k in ("mean", "log_std")}
        mask: dict[str, Tensor] = {"unit": torch.cat([d.unit_mask for d in dists], dim=0)}
        if any(d.action_mask is not None for d in dists):
            mask["action"] = torch.cat([
                d.action_mask if d.action_mask is not None
                else torch.ones(d.B, d.U, d.group.mask_size, dtype=torch.bool, device=d.unit_mask.device)
                for d in dists], dim=0)
        return UnitsDist(first.group, params, mask)
```

- [ ] **Step 4: Build `UnitsDist` in `make_distribution` and export it**

In `src/colosseum/sp2/networks/dist/tree.py`, replace:

```python
from colosseum.sp2.networks.dist.leaf import CategoricalDist, DiagGaussianDist, MultiCategoricalDist
```

with:

```python
from colosseum.sp2.networks.dist.leaf import CategoricalDist, DiagGaussianDist, MultiCategoricalDist
from colosseum.sp2.networks.dist.units import UnitsDist
```

In `src/colosseum/sp2/networks/dist/tree.py`, replace:

```python
        else:
            raise NotImplementedError("Units action groups need UnitsDist (added in T2.2)")
```

with:

```python
        else:
            parts[group.path] = UnitsDist(group, p)
```

Replace the whole content of `src/colosseum/sp2/networks/dist/__init__.py` with:

```python
"""Action distributions over decider trees (SP2 spec block 2)."""

from colosseum.sp2.networks.dist.base import Distribution
from colosseum.sp2.networks.dist.leaf import CategoricalDist, DiagGaussianDist, MultiCategoricalDist
from colosseum.sp2.networks.dist.tree import TreeDist, make_distribution
from colosseum.sp2.networks.dist.units import UnitsDist

__all__ = ["CategoricalDist", "DiagGaussianDist", "Distribution", "MultiCategoricalDist", "TreeDist", "UnitsDist",
           "make_distribution"]
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_units_distribution.py tests/unit/test_tree_distribution.py tests/unit/test_leaf_distributions.py -q`
Expected: `28 passed`.

- [ ] **Step 6: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/sp2/networks/dist/units.py \
  src/colosseum/sp2/networks/dist/tree.py \
  src/colosseum/sp2/networks/dist/__init__.py \
  tests/unit/test_units_distribution.py
git commit -m "feat(sp2): UnitsDist with per-unit deciders, masks and only_if gating"
git push origin sp2-game-model
```

---

### Task T2.3: Model protocol, base classes, `ComposedModel` with a critic encoder, heads, `NormalizeObs`, test models

Spec block 3 and the model parts of block 2 ("`NormalizeObs` знает путь своего листа", "Хелперы голов").

- `PolicyModel`: `step(obs, state, action_mask) -> PolicyStep(dist, state)` is the policy path (workers and eval compute no values); `unroll(obs, state0, reset_after, action_mask, global_state, with_value) -> UnrollOutput(dist, value)` is the learner path over chunk slots, time-major `[S*B]`; `with_value=False` returns `value=None` and needs no global state. `unroll` is abstract: there is no generic loop helper for monolithic models. `update_normalizers(obs, global_state)` feeds every `NormalizeObs` the leaf at its path of its source tree (a global-state normalizer is skipped when `global_state` is None).
- `act()` samples (or takes the mode) and returns the action tree, the joint log-prob `[B]` and the per-decider log-probs `[B, K]`.
- `ComposedModel(encoder, core, policy, value, critic_encoder=None)`: the encoder returns a latent or `EncoderOutput(latent, aux)`; `aux` goes around the core straight to `policy(features, aux)` (unit embeddings, spatial maps); the value head sees `features ⊕ critic_encoder(global_state)` when a critic encoder is set (`unroll(with_value=True)` without `global_state` is then a `ValueError`). Cores are SP1's (`colosseum.networks.cores`, unchanged).
- Heads: `UnitsHead` (unit features `[B, U, F]` -> `UnitsDist` parameters) and `gridnet_to_units` (`[B, C, H, W]` -> `[B, H*W, C]`); `heads.make_distribution` re-exports T2.1's.
- `NormalizeObs(shape, path=(), source="obs" | "global_state", eps, clip)` with SP1's running-statistics math; any input dtype (a `uint8` leaf is cast to float inside).
- The shared kit gets generic test models for ANY role (`make_test_model`, `CORE_KINDS`, `RandomPolicy` and their parts), used by every later contract test.

The contract test of this task (`test_unroll_reproduces_step_per_decider`) proves for all four cores, on a role with Dict observations (a `uint8` grid) and a `Units` action with `only_if`, that `unroll` gives the per-decider log-probs of consecutive `step` calls with the same resets.

**Files:**
- Create: `src/colosseum/sp2/networks/base.py`, `model.py`, `composed.py`, `heads.py`, `normalization.py`
- Modify: `tests/game_helpers.py` (import block; append the tiny-models section)
- Test: `tests/unit/test_policy_model_v2.py`, `tests/unit/test_units_head.py`, `tests/unit/test_normalize_obs_tree.py`

**Interfaces:**
- Consumes: `Tree`, `tree_get`, `tree_leaves`, `tree_map` (T1.1); `ActionSpec`, `ObsSpec` (T1.3); `Distribution`, `make_distribution`, `UnitsDist` (T2.1, T2.2); `RoleSpec` and toy games (T1.4); SP1 `colosseum.networks.cores` (`Core`, `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`) and `colosseum.networks.state` (`State`, `batch_size_of`, `tree_leaves`, `where_done`), unchanged.
- Produces:
  - `colosseum.sp2.networks.base`: `EncoderOutput`, `BaseEncoder`, `BasePolicy`, `BaseValue`, `BaseCriticEncoder`;
  - `colosseum.sp2.networks.model`: `PolicyStep`, `UnrollOutput`, `ActOutput`, `PolicyModel`, `act` (addition: `PolicyModel._check_unroll_args(num_slots, state0)` for subclasses, SP1 semantics);
  - `colosseum.sp2.networks.composed.ComposedModel` (submodules `encoder`, `core`, `policy`, `value`, `critic_encoder`);
  - `colosseum.sp2.networks.heads`: `UnitsHead(group, in_dim, hidden=0)`, `gridnet_to_units(x)`, `make_distribution`;
  - `colosseum.sp2.networks.normalization`: `NormalizeObs` (attributes `shape`, `path`, `source`, submodule `rms`), `RunningMeanStd`;
  - `tests/game_helpers.py`: `CORE_KINDS`, `make_core(kind, input_dim, hidden=16)`, `flatten_tree(spec, tree)`, `params_tree(spec, per_group)`, `GenericEncoder(observation_space, hidden=16)` (records `seen_dtypes`), `GenericCriticEncoder(global_state_space, hidden=16)`, `TreePolicyHead(in_dim, action_spec, hidden=16)`, `GenericValue(in_dim, hidden=16)`, `make_test_model(role, core="none", hidden=16) -> ComposedModel` (a critic encoder when the role declares a global state), `RandomPolicy(role)` (stateless; uniform over legal discrete values, N(0, 1) for box parts; `unroll` values are zeros).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_policy_model_v2.py`:

```python
"""PolicyModel v2: act() with per-decider log-probs, unroll == step for 4 cores, critic, aux, normalizers (T2.3)."""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from colosseum.networks.cores import LSTMCore, NoCore
from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import tree_index, tree_map, tree_stack, tree_to_torch
from colosseum.sp2.networks.base import BaseEncoder, BasePolicy, EncoderOutput
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.dist import make_distribution
from colosseum.sp2.networks.heads import UnitsHead
from colosseum.sp2.networks.model import act
from colosseum.sp2.networks.normalization import NormalizeObs
from game_helpers import (
    CORE_KINDS,
    GenericValue,
    GlobalStateGame,
    RandomPolicy,
    TreePolicyHead,
    TurnTakingGame,
    UnitsGame,
    make_test_model,
)

UNITS_ROLE = UnitsGame(max_units=3).spec.roles["player"]
S, B = 5, 3


def _slots(role, seed=0):
    """Random observations [S, B, ...] and masks with some absent units, as torch trees."""
    space = role.observation_space
    space.seed(seed)
    obs = tree_stack([tree_stack([space.sample() for _ in range(B)]) for _ in range(S)])
    mask = ActionSpec.from_space(role.action_space).full_mask((S, B))
    rng = np.random.default_rng(seed)
    mask["units"]["unit"][...] = rng.random((S, B, 3)) < 0.7
    return tree_to_torch(obs), tree_to_torch(mask)


def _reset_after():
    reset_after = torch.zeros(S, B, dtype=torch.bool)
    reset_after[1, 0] = reset_after[3, 2] = True
    return reset_after


def _step_through(model, obs, mask, reset_after):
    state = model.initial_state(B)
    actions, unit_lps, joint = [], [], []
    for s in range(S):
        out = act(model, tree_index(obs, s), state, tree_index(mask, s))
        actions.append(out.actions)
        unit_lps.append(out.unit_log_probs)
        joint.append(out.log_probs)
        state = model.reset_state(out.state, reset_after[s])
    flat_actions = tree_map(lambda *xs: torch.cat(xs, dim=0), actions[0], *actions[1:])
    return flat_actions, torch.cat(unit_lps), torch.cat(joint)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_unroll_reproduces_step_per_decider(core):
    torch.manual_seed(0)
    model = make_test_model(UNITS_ROLE, core=core)
    obs, mask = _slots(UNITS_ROLE)
    reset_after = _reset_after()
    actions, unit_lps, joint = _step_through(model, obs, mask, reset_after)
    assert unit_lps.shape == (S * B, 1 + 3)                         # decider 0 (base) + 3 units
    assert torch.allclose(joint, unit_lps.sum(-1))
    out = model.unroll(obs, model.initial_state(B), reset_after, mask, with_value=True)
    assert torch.allclose(out.dist.unit_log_prob(actions), unit_lps, atol=1e-5)
    assert out.value.shape == (S * B,)
    assert model.unroll(obs, model.initial_state(B), reset_after, mask, with_value=False).value is None


def test_act_returns_env_format_actions():
    model = make_test_model(UNITS_ROLE)
    obs, mask = _slots(UNITS_ROLE)
    out = act(model, tree_index(obs, 0), None, tree_index(mask, 0), deterministic=True)
    assert out.actions["base"].shape == (B,) and out.actions["units"]["move"].shape == (B, 3)
    assert out.unit_log_probs.shape == (B, 4) and out.state is None
    absent = ~tree_index(mask, 0)["units"]["unit"]
    assert torch.all(out.unit_log_probs[:, 1:][absent] == 0)


def test_unroll_argument_errors():
    model = make_test_model(UNITS_ROLE, core="lstm")
    obs, mask = _slots(UNITS_ROLE)
    with pytest.raises(ValueError, match="state0 is None"):
        model.unroll(obs, None, _reset_after(), mask)
    assert model.is_stateful and not make_test_model(UNITS_ROLE).is_stateful


def test_critic_encoder_feeds_the_value_only():
    role = GlobalStateGame().spec.roles["player"]
    model = make_test_model(role, core="gru")
    assert model.critic_encoder is not None
    obs = torch.randn(S, B, 2)
    reset_after = torch.zeros(S, B, dtype=torch.bool)
    with pytest.raises(ValueError, match="needs global_state"):
        model.unroll(obs, model.initial_state(B), reset_after)
    gs_a, gs_b = torch.zeros(S, B, 4), torch.ones(S, B, 4)
    a = model.unroll(obs, model.initial_state(B), reset_after, global_state=gs_a)
    b = model.unroll(obs, model.initial_state(B), reset_after, global_state=gs_b)
    actions = torch.zeros(S * B, dtype=torch.long)
    assert torch.allclose(a.dist.log_prob(actions), b.dist.log_prob(actions))
    assert not torch.allclose(a.value, b.value)
    assert model.unroll(obs, model.initial_state(B), reset_after, with_value=False).value is None


class _EntityEncoder(BaseEncoder):
    """Latent from the grid; per-unit embeddings of the entity list go around the core in ``aux``."""

    def __init__(self, hidden=16, emb=8):
        super().__init__()
        self.grid = nn.Linear(16, hidden)
        self.entities = nn.Linear(3, emb)

    @property
    def latent_dim(self):
        return self.grid.out_features

    def forward(self, obs):
        latent = torch.relu(self.grid(obs["grid"].float().flatten(1) / 255.0))
        return EncoderOutput(latent, {"units": torch.relu(self.entities(obs["entities"]))})


class _AuxPolicy(BasePolicy):
    def __init__(self, in_dim, spec, emb=8):
        super().__init__()
        self.spec = spec
        self.base = nn.Linear(in_dim, 3)
        self.units = UnitsHead(spec.groups[1], in_dim + emb, hidden=8)

    def forward(self, features, aux):
        per_unit = torch.cat([features.unsqueeze(1).expand(-1, aux["units"].shape[1], -1), aux["units"]], dim=-1)
        return make_distribution(self.spec, {"base": self.base(features), "units": self.units(per_unit)})


def test_encoder_aux_bypasses_the_core():
    torch.manual_seed(0)
    spec = ActionSpec.from_space(UNITS_ROLE.action_space)
    encoder = _EntityEncoder()
    core = LSTMCore(16, hidden_size=16)
    model = ComposedModel(encoder, core, _AuxPolicy(16, spec), GenericValue(16))
    obs, mask = _slots(UNITS_ROLE)
    reset_after = _reset_after()
    actions, unit_lps, _ = _step_through(model, obs, mask, reset_after)
    out = model.unroll(obs, model.initial_state(B), reset_after, mask)
    assert torch.allclose(out.dist.unit_log_prob(actions), unit_lps, atol=1e-5)
    out.dist.log_prob(actions).sum().backward()
    assert encoder.entities.weight.grad is not None and encoder.entities.weight.grad.abs().sum() > 0


def test_uint8_leaves_reach_the_encoder_unchanged():
    model = make_test_model(UNITS_ROLE)
    obs, mask = _slots(UNITS_ROLE)
    model.step(tree_index(obs, 0), None, tree_index(mask, 0))
    assert model.encoder.seen_dtypes == [torch.uint8, torch.float32, torch.int8]


class _NormEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.norm = NormalizeObs(shape=(3,), path=("entities",))
        self.gs_norm = NormalizeObs(shape=(4,), source="global_state")
        self.fc = nn.Linear(3, 4)

    @property
    def latent_dim(self):
        return 4

    def forward(self, obs):
        return self.fc(self.norm(obs["entities"]).mean(dim=1))


def test_update_normalizers_reads_each_leaf_path_and_source():
    encoder = _NormEncoder()
    model = ComposedModel(encoder, NoCore(4), TreePolicyHead(4, ActionSpec.from_space(UNITS_ROLE.action_space)),
                          GenericValue(4))
    obs = {"grid": torch.zeros(6, 4, 4, dtype=torch.uint8), "entities": torch.full((6, 3, 3), 2.0),
           "entity_mask": torch.ones(6, 3, dtype=torch.int8)}
    model.update_normalizers(obs)
    assert torch.allclose(encoder.norm.rms.mean, torch.full((3,), 2.0), atol=1e-3)
    assert torch.allclose(encoder.gs_norm.rms.mean, torch.zeros(4))         # no global_state given: skipped
    model.update_normalizers(obs, global_state=torch.full((5, 4), 3.0))
    assert torch.allclose(encoder.gs_norm.rms.mean, torch.full((4,), 3.0), atol=1e-3)


def test_random_policy_is_uniform_over_legal_actions():
    role = TurnTakingGame().spec.roles["player"]
    policy = RandomPolicy(role)
    mask = torch.tensor([[True, True, False]] * 4)
    out = act(policy, torch.zeros(4, 3), None, mask)
    assert set(out.actions.tolist()) <= {0, 1}
    assert torch.allclose(out.log_probs, torch.full((4,), -math.log(2)))
    obs, mask = _slots(UNITS_ROLE)
    units = act(RandomPolicy(UNITS_ROLE), tree_index(obs, 0), None, tree_index(mask, 0))
    assert units.unit_log_probs.shape == (B, 4) and units.actions["units"]["target"].shape == (B, 3)
    unrolled = RandomPolicy(UNITS_ROLE).unroll(obs, None, _reset_after(), mask)
    assert unrolled.value.shape == (S * B,) and torch.all(unrolled.value == 0)
```

Create `tests/unit/test_units_head.py`:

```python
"""UnitsHead and the GridNet layout helper (SP2 T2.3)."""
import torch
from gymnasium.spaces import Box, Dict, Discrete

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.dist import UnitsDist
from colosseum.sp2.networks.heads import UnitsHead, gridnet_to_units


def test_units_head_parameters_per_component():
    group = ActionSpec.from_space(Units(5, Dict([("move", Discrete(4)), ("thrust", Box(-1.0, 1.0, (2,)))]))).groups[0]
    head = UnitsHead(group, in_dim=6, hidden=8)
    params = head(torch.randn(3, 5, 6))
    assert list(params) == ["move", "thrust"]
    assert params["move"].shape == (3, 5, 4)
    assert params["thrust"]["mean"].shape == (3, 5, 2) and params["thrust"]["log_std"].shape == (2,)
    dist = UnitsDist(group, params)
    assert dist.sample()["thrust"].shape == (3, 5, 2)


def test_gridnet_to_units_is_row_major_cells():
    x = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)   # [B, C, H, W]
    units = gridnet_to_units(x)
    assert units.shape == (2, 8, 3)
    assert torch.equal(units[1, 1 * 4 + 2], x[1, :, 1, 2])
```

Create `tests/unit/test_normalize_obs_tree.py`:

```python
"""NormalizeObs v2: leaf path, source, any input dtype (SP2 T2.3)."""
import pytest
import torch

from colosseum.sp2.networks.normalization import NormalizeObs


def test_uint8_input_is_normalized_to_float():
    norm = NormalizeObs(shape=(2,), path=("grid",))
    norm.update(torch.tensor([[0, 10], [20, 30]], dtype=torch.uint8))
    out = norm(torch.tensor([[10, 20]], dtype=torch.uint8))
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.zeros(1, 2), atol=1e-3)
    assert norm.path == ("grid",) and norm.source == "obs"


def test_update_takes_any_leading_dims_and_checks_trailing_dims():
    norm = NormalizeObs(shape=(3,), source="global_state")
    norm.update(torch.ones(4, 5, 3))
    assert torch.allclose(norm.rms.mean, torch.ones(3), atol=1e-3)
    with pytest.raises(ValueError, match=r"trailing dims must equal \(3,\)"):
        norm.update(torch.ones(4, 2))
    with pytest.raises(ValueError, match="source"):
        NormalizeObs(shape=(3,), source="state")


def test_statistics_are_buffers():
    keys = set(NormalizeObs(shape=(2,)).state_dict())
    assert keys == {"rms.mean", "rms.var", "rms.count"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_policy_model_v2.py tests/unit/test_units_head.py tests/unit/test_normalize_obs_tree.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.networks.base'` (and `...networks.heads`, `...networks.normalization`).

- [ ] **Step 3: Implement the base classes and `NormalizeObs`**

Create `src/colosseum/sp2/networks/base.py`:

```python
"""Base classes for the parts of a ``ComposedModel`` (SP2 spec block 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import torch.nn as nn
from torch import Tensor

from colosseum.sp2.core.tree import Tree

if TYPE_CHECKING:
    from colosseum.sp2.networks.dist import Distribution


class EncoderOutput(NamedTuple):
    latent: Tensor                 # [B, D]: goes through the core
    aux: dict[str, Tensor]         # bypasses the core, e.g. unit embeddings [B, U, E] or spatial maps


class BaseEncoder(nn.Module, ABC):
    """Observation tree (torch, batch first, env dtypes) -> latent ``[B, D]`` (or an ``EncoderOutput``)."""

    @abstractmethod
    def forward(self, obs: Tree) -> Tensor | EncoderOutput: ...

    @property
    @abstractmethod
    def latent_dim(self) -> int: ...


class BasePolicy(nn.Module, ABC):
    """Core features ``[B, F]`` plus the encoder's ``aux`` -> action distribution (batch ``[B]``)."""

    @abstractmethod
    def forward(self, features: Tensor, aux: dict[str, Tensor]) -> Distribution: ...


class BaseValue(nn.Module, ABC):
    """Value input ``[B, F (+ G)]`` -> values ``[B]``."""

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor: ...


class BaseCriticEncoder(nn.Module, ABC):
    """Global-state tree -> ``[B, G]`` features for the value head only (centralized critic)."""

    @abstractmethod
    def forward(self, global_state: Tree) -> Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
```

Create `src/colosseum/sp2/networks/normalization.py`:

```python
"""Observation normalization (running mean/std) for one leaf of the observation or global-state tree.

``NormalizeObs`` keeps its statistics in registered buffers, so they are part of the model
``state_dict`` (weight sync, checkpoints). They change only in :meth:`NormalizeObs.update`;
``forward`` is a pure function of the current statistics. The learner calls
``PolicyModel.update_normalizers(obs, global_state)`` once per train step; the default
implementation feeds every ``NormalizeObs`` the leaf at its ``path`` of its ``source`` tree.

Usage inside an encoder::

    self.norm = NormalizeObs(shape=(F,), path=("entities",))
    ...
    x = self.norm(obs["entities"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Welford-style running mean/variance over a feature shape, in buffers."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update stats from a batch ``x`` of shape ``[N, *shape]``."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)


class NormalizeObs(nn.Module):
    """Normalize one tree leaf by running mean/std (any input dtype; the output is float32)."""

    def __init__(self, shape: Sequence[int], path: tuple[str, ...] = (),
                 source: Literal["obs", "global_state"] = "obs", eps: float = 1e-8, clip: float = 10.0) -> None:
        super().__init__()
        if source not in ("obs", "global_state"):
            raise ValueError(f"NormalizeObs: source must be 'obs' or 'global_state', got {source!r}")
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.path: tuple[str, ...] = tuple(path)
        self.source = source
        self.rms = RunningMeanStd(self.shape)
        self._clip = clip
        self._eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Add samples ``[..., *shape]`` (any leading dims) to the statistics."""
        n = len(self.shape)
        if x.dim() < n or tuple(x.shape[x.dim() - n:]) != self.shape:
            raise ValueError(
                f"NormalizeObs(shape={self.shape}, path={self.path}) cannot update from a leaf of shape "
                f"{tuple(x.shape)}: the trailing dims must equal {self.shape}. If it normalizes a transformed "
                f"leaf, override PolicyModel.update_normalizers."
            )
        flat = x.reshape(-1, *self.shape).to(self.rms.mean.dtype)
        if flat.shape[0]:
            self.rms.update(flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = (x.to(self.rms.mean.dtype) - self.rms.mean) / torch.sqrt(self.rms.var + self._eps)
        if self._clip > 0:
            normed = torch.clamp(normed, -self._clip, self._clip)
        return normed
```

- [ ] **Step 4: Implement the model protocol and `ComposedModel`**

Create `src/colosseum/sp2/networks/model.py`:

```python
"""``PolicyModel``: the model protocol of workers (``step``), learners (``unroll``) and eval (SP2 block 3).

- ``step(obs, state, action_mask)`` -> ``PolicyStep(dist, state)``: the policy path only. Workers
  and eval never compute values.
- ``unroll(obs, state0, reset_after, action_mask, global_state, with_value)`` ->
  ``UnrollOutput(dist, value)``: the learner path over chunk slots ``[S, B, ...]``, time-major
  flattened ``[S*B]`` (index ``s*B + b``). ``reset_after[s, b]`` resets the state AFTER slot s.
  ``with_value=False`` (BC, kickstart teacher) returns ``value=None`` and needs no global state.
- Contract: ``unroll`` gives the same policy distributions as consecutive ``step`` calls with
  the same resets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.networks.state import State, batch_size_of, tree_leaves, where_done
from colosseum.sp2.core.tree import Tree, tree_get
from colosseum.sp2.networks.dist import Distribution
from colosseum.sp2.networks.normalization import NormalizeObs


class PolicyStep(NamedTuple):
    dist: Distribution             # batch [B]
    state: State


class UnrollOutput(NamedTuple):
    dist: Distribution             # batch [S*B], time-major (index s*B + b)
    value: Tensor | None           # [S*B]; None when with_value=False


class ActOutput(NamedTuple):
    actions: Tree                  # torch, batch [B]
    log_probs: Tensor              # [B]
    unit_log_probs: Tensor         # [B, K]
    state: State


class PolicyModel(nn.Module, ABC):
    """Actor-critic with an optional recurrent/memory state (an opaque pytree, batch first)."""

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        """State at the start of an episode. Stateless models return None."""
        return None

    @abstractmethod
    def step(self, obs: Tree, state: State, action_mask: Tree | None = None) -> PolicyStep:
        """One decision for a batch: obs leaves ``[B, ...]``; the mask (if any) is applied to the dist."""

    @abstractmethod
    def unroll(self, obs: Tree, state0: State, reset_after: Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput:
        """Slots ``[S, B, ...]``; ``reset_after [S, B]`` bool; masks ``[S, B, ...]``."""

    def _check_unroll_args(self, num_slots: int, state0: State) -> None:
        if num_slots < 1:
            raise ValueError(f"unroll needs at least one slot, got S={num_slots}")
        if state0 is None and self.is_stateful:
            raise ValueError(f"unroll: state0 is None but {type(self).__name__} is stateful; pass "
                             f"initial_state(B) or the stored state of the sequence start")

    def reset_state(self, state: State, done: Tensor) -> State:
        """Replace the rows of ``state`` where ``done`` ([B] bool) with initial-state rows."""
        batch = batch_size_of(state)
        if batch is None:
            return state
        device = tree_leaves(state)[0].device
        return where_done(done, self.initial_state(batch, device), state)

    @torch.no_grad()
    def update_normalizers(self, obs: Tree, global_state: Tree | None = None) -> None:
        """Update every ``NormalizeObs`` submodule from the leaf at its path of its source tree.

        Called once per train step with all fresh observations (leaves ``[N, ...]``). A
        global-state normalizer is skipped when ``global_state`` is None.
        """
        for module in self.modules():
            if isinstance(module, NormalizeObs):
                source = obs if module.source == "obs" else global_state
                if source is not None:
                    module.update(tree_get(source, module.path))

    @property
    def is_stateful(self) -> bool:
        return self.initial_state(1) is not None


@torch.no_grad()
def act(model: PolicyModel, obs: Tree, state: State, action_mask: Tree | None = None,
        deterministic: bool = False) -> ActOutput:
    """Inference helper: step the model, then sample (or take the mode) and score the action."""
    out = model.step(obs, state, action_mask)
    actions = out.dist.mode() if deterministic else out.dist.sample()
    unit_log_probs = out.dist.unit_log_prob(actions)
    return ActOutput(actions=actions, log_probs=out.dist.log_prob(actions), unit_log_probs=unit_log_probs,
                     state=out.state)
```

Create `src/colosseum/sp2/networks/composed.py`:

```python
"""``ComposedModel``: encoder -> core -> policy head / value head, with an optional critic encoder."""

from __future__ import annotations

import torch
from torch import Tensor

from colosseum.networks.cores import Core
from colosseum.networks.state import State
from colosseum.sp2.core.tree import Tree, tree_leaves, tree_map
from colosseum.sp2.networks.base import BaseCriticEncoder, BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.sp2.networks.model import PolicyModel, PolicyStep, UnrollOutput


def _encode(encoder: BaseEncoder, obs: Tree) -> EncoderOutput:
    out = encoder(obs)
    return out if isinstance(out, EncoderOutput) else EncoderOutput(out, {})


def _flatten_time(tree: Tree, num_slots: int, batch: int) -> Tree:
    return tree_map(lambda x: x.reshape(num_slots * batch, *x.shape[2:]), tree)


class ComposedModel(PolicyModel):
    """``encoder(obs) -> core -> policy(features, aux)``; ``value(features ⊕ critic_encoder(global_state))``.

    Submodules: ``encoder``, ``core``, ``policy``, ``value`` and ``critic_encoder`` (None if unused).
    """

    def __init__(self, encoder: BaseEncoder, core: Core, policy_head: BasePolicy, value_head: BaseValue,
                 critic_encoder: BaseCriticEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.policy = policy_head
        self.value = value_head
        self.critic_encoder = critic_encoder

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return self.core.initial_state(batch_size, device)

    def step(self, obs: Tree, state: State, action_mask: Tree | None = None) -> PolicyStep:
        enc = _encode(self.encoder, obs)
        features, next_state = self.core.step(enc.latent, state)
        dist = self.policy(features, enc.aux)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return PolicyStep(dist=dist, state=next_state)

    def unroll(self, obs: Tree, state0: State, reset_after: Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput:
        S, B = (int(d) for d in tree_leaves(obs)[0].shape[:2])
        self._check_unroll_args(S, state0)
        if with_value and self.critic_encoder is not None and global_state is None:
            raise ValueError("unroll(with_value=True) of a model with a critic_encoder needs global_state")
        enc = _encode(self.encoder, _flatten_time(obs, S, B))
        latent = enc.latent.reshape(S, B, -1)
        features = self.core.unroll(latent, state0, reset_after).reshape(S * B, -1)
        dist = self.policy(features, enc.aux)
        if action_mask is not None:
            dist = dist.apply_mask(_flatten_time(action_mask, S, B))
        if not with_value:
            return UnrollOutput(dist=dist, value=None)
        value_in = features
        if self.critic_encoder is not None:
            value_in = torch.cat([features, self.critic_encoder(_flatten_time(global_state, S, B))], dim=-1)
        return UnrollOutput(dist=dist, value=self.value(value_in))

    def reset_state(self, state: State, done: Tensor) -> State:
        return self.core.reset_state(state, done)
```

- [ ] **Step 5: Implement the head helpers**

Create `src/colosseum/sp2/networks/heads.py`:

```python
"""Policy-head helpers: ``UnitsHead`` (per-unit features -> ``UnitsDist`` parameters) and GridNet layout."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.sp2.core.specs import ActionGroup
from colosseum.sp2.networks.dist import make_distribution

__all__ = ["UnitsHead", "gridnet_to_units", "make_distribution"]


class UnitsHead(nn.Module):
    """Unit features ``[B, U, F]`` -> parameters of one units group: discrete component -> logits
    ``[B, U, n]``; box component -> ``{"mean": [B, U, d], "log_std": [d]}`` (a learned parameter).
    ``hidden > 0`` adds a shared ``Linear + ReLU`` layer per unit first."""

    def __init__(self, group: ActionGroup, in_dim: int, hidden: int = 0) -> None:
        super().__init__()
        if group.kind != "units":
            raise ValueError(f"UnitsHead needs a units action group, got {group.kind!r}")
        self.group = group
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU()) if hidden > 0 else nn.Identity()
        width = hidden if hidden > 0 else in_dim
        self.heads = nn.ModuleDict()
        self.log_std = nn.ParameterDict()
        for c in group.units.components:
            self.heads[c.name] = nn.Linear(width, c.size)
            if c.kind == "box":
                self.log_std[c.name] = nn.Parameter(torch.zeros(c.size))

    def forward(self, unit_features: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        h = self.trunk(unit_features)
        out: dict[str, Tensor | dict[str, Tensor]] = {}
        for c in self.group.units.components:
            y = self.heads[c.name](h)
            out[c.name] = y if c.kind == "discrete" else {"mean": y, "log_std": self.log_std[c.name]}
        return out


def gridnet_to_units(x: Tensor) -> Tensor:
    """``[B, C, H, W]`` -> ``[B, H*W, C]``: every grid cell is a unit (row-major cell order)."""
    return x.flatten(2).transpose(1, 2)
```

- [ ] **Step 6: Add the test models to the shared kit**

In `tests/game_helpers.py` (the import block after `from typing import Any`), replace:

```python
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary

from colosseum.sp2.core.specs import ActionSpec
from colosseum.sp2.core.tree import Tree
from colosseum.sp2.envs.contract import EpisodeTracker
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult
from colosseum.sp2.envs.spaces import Units
```

with:

```python
import gymnasium
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary

from colosseum.networks.cores import Core, GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import Tree, tree_get, tree_leaves, tree_map
from colosseum.sp2.envs.contract import EpisodeTracker
from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, Outcome, RoleSpec, SeatSpec, StepResult
from colosseum.sp2.envs.spaces import Units
from colosseum.sp2.networks.base import BaseCriticEncoder, BaseEncoder, BasePolicy, BaseValue, EncoderOutput
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.dist import Distribution, make_distribution
from colosseum.sp2.networks.heads import UnitsHead
from colosseum.sp2.networks.model import PolicyModel, PolicyStep, UnrollOutput
```

Append to the end of `tests/game_helpers.py`:

```python


# ---------------------------------------------------------------------------
# Tiny models (T2.3)
# ---------------------------------------------------------------------------

CORE_KINDS = ("none", "lstm", "gru", "attention")


def make_core(kind: str, input_dim: int, hidden: int = 16) -> Core:
    """A core of ``kind`` (one of ``CORE_KINDS``) sized for tests."""
    if kind == "none":
        return NoCore(input_dim)
    if kind == "lstm":
        return LSTMCore(input_dim, hidden_size=hidden)
    if kind == "gru":
        return GRUCore(input_dim, hidden_size=hidden)
    if kind == "attention":
        return WindowAttentionCore(input_dim, d_model=hidden, window=4, num_heads=2)
    raise ValueError(f"unknown core kind {kind!r}; use one of {CORE_KINDS}")


def flatten_tree(spec: ObsSpec, tree: Tree) -> torch.Tensor:
    """Every leaf ``[B, ...]`` cast to float and flattened, concatenated in leaf order -> ``[B, F]``."""
    parts = []
    for leaf in spec.leaves:
        x = tree_get(tree, leaf.path) if spec.is_dict else tree
        parts.append(x.reshape(x.shape[0], -1).float())
    return torch.cat(parts, dim=-1)


def params_tree(spec: ActionSpec, per_group: dict[tuple[str, ...], Any]) -> Tree:
    """Per-group distribution parameters (keyed by group path) as the tree ``make_distribution`` takes."""
    if not spec.is_dict:
        return per_group[spec.groups[0].path]
    tree: dict = {}
    for path, value in per_group.items():
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return tree


def _flat_size(spec: ObsSpec) -> int:
    return sum(int(np.prod(leaf.shape)) if leaf.shape else 1 for leaf in spec.leaves)


class GenericEncoder(BaseEncoder):
    """Any observation space: flatten every leaf (cast to float) -> Linear -> ReLU -> ``[B, hidden]``.

    Records the dtypes of the leaves it was last called with in ``seen_dtypes``.
    """

    def __init__(self, observation_space: gymnasium.Space, hidden: int = 16) -> None:
        super().__init__()
        self.obs_spec = ObsSpec.from_space(observation_space)
        self.fc = nn.Linear(_flat_size(self.obs_spec), hidden)
        self._latent_dim = hidden
        self.seen_dtypes: list[torch.dtype] = []

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs: Tree) -> EncoderOutput:
        self.seen_dtypes = [leaf.dtype for leaf in tree_leaves(obs)]
        return EncoderOutput(torch.relu(self.fc(flatten_tree(self.obs_spec, obs))), {})


class GenericCriticEncoder(BaseCriticEncoder):
    """Any global-state space: flatten -> Linear -> ReLU -> ``[B, hidden]``."""

    def __init__(self, global_state_space: gymnasium.Space, hidden: int = 16) -> None:
        super().__init__()
        self.gs_spec = ObsSpec.from_space(global_state_space)
        self.fc = nn.Linear(_flat_size(self.gs_spec), hidden)
        self._out = hidden

    @property
    def output_dim(self) -> int:
        return self._out

    def forward(self, global_state: Tree) -> torch.Tensor:
        return torch.relu(self.fc(flatten_tree(self.gs_spec, global_state)))


class TreePolicyHead(BasePolicy):
    """Any action space: a linear head per group; a units group gets the features concatenated with a
    learned per-slot embedding of size ``hidden``, fed to a ``UnitsHead``."""

    def __init__(self, in_dim: int, action_spec: ActionSpec, hidden: int = 16) -> None:
        super().__init__()
        self.spec = action_spec
        self.heads = nn.ModuleDict()
        self.log_std = nn.ParameterDict()
        self.slots = nn.ParameterDict()
        for i, group in enumerate(action_spec.groups):
            key = str(i)
            if group.kind == "units":
                self.slots[key] = nn.Parameter(torch.randn(group.units.max_units, hidden) * 0.5)
                self.heads[key] = UnitsHead(group, in_dim + hidden)
            elif group.kind == "box":
                self.heads[key] = nn.Linear(in_dim, group.box_dim)
                self.log_std[key] = nn.Parameter(torch.zeros(group.box_dim))
            else:
                self.heads[key] = nn.Linear(in_dim, group.mask_size)

    def forward(self, features: torch.Tensor, aux: dict[str, torch.Tensor]) -> Distribution:
        params: dict[tuple[str, ...], Any] = {}
        for i, group in enumerate(self.spec.groups):
            key = str(i)
            if group.kind == "units":
                slots = self.slots[key].unsqueeze(0).expand(features.shape[0], -1, -1)
                per_unit = features.unsqueeze(1).expand(-1, slots.shape[1], -1)
                params[group.path] = self.heads[key](torch.cat([per_unit, slots], dim=-1))
            elif group.kind == "box":
                params[group.path] = {"mean": self.heads[key](features), "log_std": self.log_std[key]}
            else:
                params[group.path] = self.heads[key](features)
        return make_distribution(self.spec, params_tree(self.spec, params))


class GenericValue(BaseValue):
    """``[B, in_dim]`` -> Linear -> ReLU -> Linear -> ``[B]``."""

    def __init__(self, in_dim: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_test_model(role: RoleSpec, core: str = "none", hidden: int = 16) -> ComposedModel:
    """A small ``ComposedModel`` for any role; a role with a global state gets a critic encoder."""
    encoder = GenericEncoder(role.observation_space, hidden)
    trunk = make_core(core, encoder.latent_dim, hidden)
    critic = GenericCriticEncoder(role.global_state_space, hidden) if role.global_state_space is not None else None
    value_in = trunk.output_dim + (critic.output_dim if critic is not None else 0)
    return ComposedModel(encoder, trunk, TreePolicyHead(trunk.output_dim, ActionSpec.from_space(role.action_space),
                                                        hidden), GenericValue(value_in, hidden), critic)


class RandomPolicy(PolicyModel):
    """Stateless: uniform over legal discrete actions (zero logits + the mask), N(0, 1) for box parts.
    ``unroll`` returns zero values (or None with ``with_value=False``)."""

    def __init__(self, role: RoleSpec) -> None:
        super().__init__()
        self.spec = ActionSpec.from_space(role.action_space)

    def _dist(self, batch: int, device: torch.device) -> Distribution:
        params: dict[tuple[str, ...], Any] = {}
        for group in self.spec.groups:
            if group.kind == "units":
                U = group.units.max_units
                params[group.path] = {
                    c.name: (torch.zeros(batch, U, c.size, device=device) if c.kind == "discrete"
                             else {"mean": torch.zeros(batch, U, c.size, device=device),
                                   "log_std": torch.zeros(c.size, device=device)})
                    for c in group.units.components}
            elif group.kind == "box":
                params[group.path] = {"mean": torch.zeros(batch, group.box_dim, device=device),
                                      "log_std": torch.zeros(group.box_dim, device=device)}
            else:
                params[group.path] = torch.zeros(batch, group.mask_size, device=device)
        return make_distribution(self.spec, params_tree(self.spec, params))

    def step(self, obs: Tree, state: Any, action_mask: Tree | None = None) -> PolicyStep:
        first = tree_leaves(obs)[0]
        dist = self._dist(int(first.shape[0]), first.device)
        return PolicyStep(dist=dist if action_mask is None else dist.apply_mask(action_mask), state=None)

    def unroll(self, obs: Tree, state0: Any, reset_after: torch.Tensor, action_mask: Tree | None = None,
               global_state: Tree | None = None, with_value: bool = True) -> UnrollOutput:
        first = tree_leaves(obs)[0]
        S, B = int(first.shape[0]), int(first.shape[1])
        dist = self._dist(S * B, first.device)
        if action_mask is not None:
            dist = dist.apply_mask(tree_map(lambda m: m.reshape(S * B, *m.shape[2:]), action_mask))
        return UnrollOutput(dist=dist, value=torch.zeros(S * B, device=first.device) if with_value else None)
```

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_policy_model_v2.py tests/unit/test_units_head.py tests/unit/test_normalize_obs_tree.py -q`
Expected: `16 passed`.

- [ ] **Step 8: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/sp2/networks/base.py \
  src/colosseum/sp2/networks/model.py \
  src/colosseum/sp2/networks/composed.py \
  src/colosseum/sp2/networks/heads.py \
  src/colosseum/sp2/networks/normalization.py \
  tests/game_helpers.py \
  tests/unit/test_policy_model_v2.py \
  tests/unit/test_units_head.py \
  tests/unit/test_normalize_obs_tree.py
git commit -m "feat(sp2): PolicyModel step/unroll protocol, ComposedModel with critic encoder, units heads"
git push origin sp2-game-model
```

---

### Task T2.4: Role resolution, `build_model`, `make_env`, `env_spec`

Spec block 5 ("Роли и агенты") and block 3 ("Конфиг: `networks.critic_encoder_class`"). An agent plays the roles listed in `agents.<id>.roles`, or every role of the game when omitted; all its roles must have equal observation, action and global-state spaces (`role_signature`), otherwise `ConfigError` with the hint "roles with different spaces need separate agents". The agent's network is built for those spaces: `build_model` passes `observation_space`, `action_space`, `global_state_space` and `action_spec` to every constructor that names them as parameters (SP1's `in_dim` probing rule), never overriding a key that `networks.kwargs` sets. `networks.critic_encoder_class` with a role that declares no global state is a `ConfigError`.

`validate_config` is NOT part of this task (T6.2 adds it to the same module).

**Files:**
- Create: `src/colosseum/sp2/core/roles.py`
- Create: `src/colosseum/sp2/core/registry.py`
- Test: `tests/unit/test_agent_roles.py`, `tests/unit/test_build_model_v2.py`

**Interfaces:**
- Consumes: `ColosseumConfig`, `agent_roles`, `get_trainable_agent_ids` (T1.7); `GameSpec`, `RoleSpec`, `MultiAgentEnv` (T1.4); `ObsSpec`, `ActionSpec` signatures (T1.3); `ComposedModel`, `PolicyModel`, `BaseCriticEncoder` (T2.3); generic test models (T2.3); SP1 `colosseum.networks.cores`.
- Produces:
  - `colosseum.sp2.core.roles`: `role_signature(role) -> str` (`"obs=<sig>|act=<sig>|gs=<sig or none>"`), `resolve_agent_roles(config, spec) -> dict[str, list[str]]`, `agent_role_spec(spec, roles) -> RoleSpec` — the contract;
  - `colosseum.sp2.core.registry`: `import_class(dotted_path)`, `make_env(config)`, `env_spec(config)`, `build_model(agent_config, role)` — the contract. `make_env` wraps any construction error in `ConfigError` and rejects non-`MultiAgentEnv` classes; `env_spec` closes the env and lets `GameSpec.validate`'s `EnvContractError` through; a `spec` attribute that is not a `GameSpec` is a `ConfigError`.

- [ ] **Step 1: Write the failing tests**

The `model_class` tests reference classes of the test module by its bare name (`test_build_model_v2.InjectedModel`): pytest imports test modules under their basename (prepend mode), and the basename is unique (conftest check).

Create `tests/unit/test_agent_roles.py`:

```python
"""Role signatures and agents.<id>.roles resolution (SP2 T2.4)."""
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import ColosseumConfig
from colosseum.sp2.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec
from game_helpers import AsymmetricGame, EliminationFFA, GlobalStateGame

ASYM = AsymmetricGame().spec
NETWORKS = {"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
            "value_class": "game_helpers.GenericValue"}


def _config(agents=None, env="game_helpers.AsymmetricGame"):
    data = {"env": {"env_class": env}, "networks": NETWORKS}
    if agents is not None:
        data["agents"] = agents
    return ColosseumConfig.model_validate(data)


def test_role_signature_names_every_space():
    hunter, prey = ASYM.roles["hunter"], ASYM.roles["prey"]
    assert role_signature(hunter) == "obs=<root>:float32[4]|act=<root>:discrete(5)|gs=none"
    assert role_signature(prey) != role_signature(hunter)
    with_gs = GlobalStateGame().spec.roles["player"]
    assert role_signature(with_gs).endswith("|gs=<root>:float32[4]")
    assert role_signature(RoleSpec(Box(-1.0, 1.0, (4,), dtype=np.float32), Discrete(5))) == role_signature(hunter)


def test_single_role_games_default_to_every_role():
    assert resolve_agent_roles(_config(env="game_helpers.EliminationFFA"), EliminationFFA().spec) == \
        {"agent_0": ["player"]}


def test_explicit_roles_for_an_asymmetric_game():
    cfg = _config({"hunter": {"roles": ["hunter"]}, "prey": {"roles": ["prey"]}})
    roles = resolve_agent_roles(cfg, ASYM)
    assert roles == {"hunter": ["hunter"], "prey": ["prey"]}
    assert agent_role_spec(ASYM, roles["prey"]) is ASYM.roles["prey"]


def test_omitted_roles_with_different_spaces_need_separate_agents():
    with pytest.raises(ConfigError, match="set agents.agent_0.roles; roles with different spaces need separate agents"):
        resolve_agent_roles(_config(), ASYM)


def test_unknown_role():
    with pytest.raises(ConfigError, match=r"agents.a.roles: unknown roles \['wolf'\]; the game has roles "
                                          r"\['hunter', 'prey'\]"):
        resolve_agent_roles(_config({"a": {"roles": ["wolf"]}}), ASYM)


def test_roles_of_one_agent_must_share_spaces():
    with pytest.raises(ConfigError, match="roles 'hunter' and 'prey' have different spaces"):
        resolve_agent_roles(_config({"a": {"roles": ["hunter", "prey"]}}), ASYM)


def test_roles_with_equal_spaces_can_share_an_agent():
    space = RoleSpec(Box(-1.0, 1.0, (3,), dtype=np.float32), Discrete(2))
    spec = GameSpec(roles={"attacker": space, "defender": space},
                    layouts={"1v1": (SeatSpec("attacker", 0), SeatSpec("defender", 1))})
    assert resolve_agent_roles(_config(), spec) == {"agent_0": ["attacker", "defender"]}
    assert resolve_agent_roles(_config({"x": {"roles": ["defender"]}}), spec) == {"x": ["defender"]}
```

Create `tests/unit/test_build_model_v2.py`:

```python
"""build_model with role injection, critic encoder, make_env / env_spec (SP2 T2.4)."""
import numpy as np
import pytest
import torch

from colosseum.core.errors import ConfigError, EnvContractError
from colosseum.networks.cores import LSTMCore
from colosseum.sp2.core.config import ColosseumConfig
from colosseum.sp2.core.registry import build_model, env_spec, import_class, make_env
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.core.tree import tree_index, tree_stack, tree_to_torch
from colosseum.sp2.envs.game import GameSpec, RoleSpec, SeatSpec
from colosseum.sp2.networks.composed import ComposedModel
from colosseum.sp2.networks.model import PolicyModel, act
from game_helpers import GlobalStateGame, RandomPolicy, SoloCounterGame, UnitsGame

UNITS_ROLE = UnitsGame(max_units=3).spec.roles["player"]
GS_ROLE = GlobalStateGame().spec.roles["player"]


def _config(networks=None, env="game_helpers.UnitsGame", env_kwargs=None):
    networks = networks or {"encoder_class": "game_helpers.GenericEncoder",
                            "policy_class": "game_helpers.TreePolicyHead",
                            "value_class": "game_helpers.GenericValue", "kwargs": {"hidden": 8}}
    return ColosseumConfig.model_validate({"env": {"env_class": env, "kwargs": env_kwargs or {}},
                                           "networks": networks})


class InjectedModel(RandomPolicy):
    """Names observation_space and action_spec (not action_space): gets exactly those two."""

    def __init__(self, observation_space, action_spec, width: int = 1) -> None:
        super().__init__(RoleSpec(observation_space, UNITS_ROLE.action_space))
        self.got = {"observation_space": observation_space, "action_spec": action_spec, "width": width}


class SpecFromKwargs(RandomPolicy):
    def __init__(self, action_spec, observation_space=None) -> None:
        super().__init__(UNITS_ROLE)
        self.got = action_spec


class NotAModel:
    def __init__(self, **kwargs) -> None:
        pass


class _CountingGame(SoloCounterGame):
    closed = 0

    def close(self) -> None:
        _CountingGame.closed += 1


class _BadTeams(SoloCounterGame):
    def __init__(self) -> None:
        super().__init__()
        self.spec = GameSpec(roles=self.spec.roles, layouts={"solo": (SeatSpec("player", 1),)})


class _NoSpec(SoloCounterGame):
    def __init__(self) -> None:
        super().__init__()
        self.spec = "solo"


def test_composed_model_gets_the_role_spaces():
    model = build_model(_config(), UNITS_ROLE)
    assert isinstance(model, ComposedModel) and model.critic_encoder is None
    assert model.encoder.obs_spec == ObsSpec.from_space(UNITS_ROLE.observation_space)
    assert model.policy.spec == ActionSpec.from_space(UNITS_ROLE.action_space)
    assert model.encoder.latent_dim == 8
    space = UNITS_ROLE.observation_space
    space.seed(0)
    obs = tree_to_torch(tree_stack([space.sample() for _ in range(2)]))
    out = act(model, obs, None, tree_to_torch(ActionSpec.from_space(UNITS_ROLE.action_space).full_mask((2,))))
    assert out.unit_log_probs.shape == (2, 4)


def test_core_and_in_dim():
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue", "kwargs": {"hidden": 8},
                   "core": {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 12}}})
    model = build_model(cfg, UNITS_ROLE)
    assert isinstance(model.core, LSTMCore) and model.core.output_dim == 12
    assert model.value.net[0].in_features == 12 and model.is_stateful


def test_critic_encoder_widens_the_value_input():
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue", "kwargs": {"hidden": 8},
                   "critic_encoder_class": "game_helpers.GenericCriticEncoder"}, env="game_helpers.GlobalStateGame")
    model = build_model(cfg, GS_ROLE)
    assert model.critic_encoder is not None and model.value.net[0].in_features == 8 + 8
    out = model.unroll(torch.zeros(3, 2, 2), None, torch.zeros(3, 2, dtype=torch.bool),
                       global_state=torch.zeros(3, 2, 4))
    assert out.value.shape == (6,)


def test_critic_encoder_needs_a_global_state():
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue",
                   "critic_encoder_class": "game_helpers.GenericCriticEncoder"})
    with pytest.raises(ConfigError, match="declares no global_state_space"):
        build_model(cfg, UNITS_ROLE)
    cfg = _config({"encoder_class": "game_helpers.GenericEncoder", "policy_class": "game_helpers.TreePolicyHead",
                   "value_class": "game_helpers.GenericValue", "critic_encoder_class": "game_helpers.GenericValue"})
    with pytest.raises(ConfigError, match="must subclass colosseum.sp2.networks.base.BaseCriticEncoder"):
        build_model(cfg, GS_ROLE)


def test_model_class_injection_by_name_only():
    model = build_model(_config({"model_class": "test_build_model_v2.InjectedModel", "kwargs": {"width": 3}}),
                        UNITS_ROLE)
    assert set(model.got) == {"observation_space", "action_spec", "width"} and model.got["width"] == 3
    assert model.got["action_spec"] == ActionSpec.from_space(UNITS_ROLE.action_space)
    assert model.got["observation_space"] is UNITS_ROLE.observation_space


def test_kwargs_win_over_injection():
    model = build_model(_config({"model_class": "test_build_model_v2.SpecFromKwargs",
                                 "kwargs": {"action_spec": "from-kwargs"}}), UNITS_ROLE)
    assert model.got == "from-kwargs"


def test_model_class_must_be_a_policy_model():
    with pytest.raises(ConfigError, match="must subclass colosseum.sp2.networks.model.PolicyModel"):
        build_model(_config({"model_class": "test_build_model_v2.NotAModel"}), UNITS_ROLE)
    assert issubclass(import_class("game_helpers.RandomPolicy"), PolicyModel)


def test_import_class_errors():
    with pytest.raises(ValueError, match="module.ClassName"):
        import_class("RandomPolicy")
    with pytest.raises(TypeError, match="not a class"):
        import_class("game_helpers.make_test_model")


def test_make_env_and_env_spec():
    env = make_env(_config(env="game_helpers.UnitsGame", env_kwargs={"max_units": 2}))
    assert isinstance(env, UnitsGame) and env.U == 2
    _CountingGame.closed = 0
    spec = env_spec(_config(env="test_build_model_v2._CountingGame"))
    assert list(spec.layouts) == ["solo"] and _CountingGame.closed == 1
    with pytest.raises(ConfigError, match="Failed to create env 'game_helpers.UnitsGame': TypeError"):
        make_env(_config(env="game_helpers.UnitsGame", env_kwargs={"bogus": 1}))
    with pytest.raises(ConfigError, match="must subclass colosseum.sp2.envs.game.MultiAgentEnv"):
        make_env(_config(env="collections.OrderedDict"))
    with pytest.raises(EnvContractError, match="team numbers must be exactly"):
        env_spec(_config(env="test_build_model_v2._BadTeams"))
    with pytest.raises(ConfigError, match="spec must be a colosseum.sp2.envs.game.GameSpec"):
        env_spec(_config(env="test_build_model_v2._NoSpec"))


def test_built_models_step_on_env_observations():
    env = UnitsGame(max_units=3)
    first = env.reset(None, "solo")
    obs = tree_to_torch(tree_stack([first.obs[0]]))
    model = build_model(_config(), UNITS_ROLE)
    mask = tree_to_torch(tree_stack([ActionSpec.from_space(UNITS_ROLE.action_space)
                                     .normalize_mask(first.action_masks[0], "seat 0")]))
    out = act(model, obs, None, mask)
    assert out.unit_log_probs[0, 2:].tolist() == [0.0, 0.0]          # units 1, 2 do not exist yet
    assert np.isfinite(out.log_probs.numpy()).all()
    assert tree_index(obs, 0)["grid"].dtype == torch.uint8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_agent_roles.py tests/unit/test_build_model_v2.py -q`
Expected: collection errors `ModuleNotFoundError: No module named 'colosseum.sp2.core.roles'` / `...registry`.

- [ ] **Step 3: Implement role resolution**

Create `src/colosseum/sp2/core/roles.py`:

```python
"""Agents and roles (SP2 spec block 5, "Роли и агенты").

An agent plays one or more roles; all its roles must have the same observation, action
and global-state spaces (one network serves them). ``agents.<id>.roles`` may be omitted
when every role of the game has the same spaces: the agent then plays every role.
"""

from __future__ import annotations

from collections.abc import Sequence

from colosseum.core.errors import ConfigError
from colosseum.sp2.core.config import ColosseumConfig
from colosseum.sp2.core.specs import ActionSpec, ObsSpec
from colosseum.sp2.envs.game import GameSpec, RoleSpec


def role_signature(role: RoleSpec) -> str:
    """Identity of a role's spaces: observation, action and global-state signatures."""
    gs = "none" if role.global_state_space is None else ObsSpec.from_space(role.global_state_space).signature()
    return (f"obs={ObsSpec.from_space(role.observation_space).signature()}"
            f"|act={ActionSpec.from_space(role.action_space).signature()}|gs={gs}")


def resolve_agent_roles(config: ColosseumConfig, spec: GameSpec) -> dict[str, list[str]]:
    """``{agent_id: roles}`` for every trainable agent; ConfigError with a fix hint on any mismatch."""
    signatures = {name: role_signature(role) for name, role in spec.roles.items()}
    out: dict[str, list[str]] = {}
    for agent_id in config.get_trainable_agent_ids():
        roles = config.agent_roles(agent_id)
        if roles is None:
            roles = list(spec.roles)
            if len({signatures[r] for r in roles}) > 1:
                raise ConfigError(
                    f"agent {agent_id!r} plays every role by default, but the roles {roles} have different "
                    f"spaces: set agents.{agent_id}.roles; roles with different spaces need separate agents"
                )
        else:
            unknown = [r for r in roles if r not in spec.roles]
            if unknown:
                raise ConfigError(f"agents.{agent_id}.roles: unknown roles {unknown}; the game has roles "
                                  f"{list(spec.roles)}")
            first = roles[0]
            for r in roles[1:]:
                if signatures[r] != signatures[first]:
                    raise ConfigError(
                        f"agents.{agent_id}.roles: roles {first!r} and {r!r} have different spaces "
                        f"({signatures[first]} vs {signatures[r]}); roles with different spaces need separate agents"
                    )
        out[agent_id] = list(roles)
    return out


def agent_role_spec(spec: GameSpec, roles: Sequence[str]) -> RoleSpec:
    """The spaces an agent's network is built for (all its roles share them)."""
    return spec.roles[roles[0]]
```

- [ ] **Step 4: Implement the registry**

Create `src/colosseum/sp2/core/registry.py`:

```python
"""Dynamic import, env construction and model building for config v2 (SP2 T2.4).

``build_model`` injects the role's spaces into the classes that ask for them by name:
``observation_space``, ``action_space``, ``global_state_space`` and ``action_spec`` are
passed to a constructor only if its signature names that parameter explicitly (the same
probing rule as SP1's ``in_dim``), and never when ``networks.kwargs`` already sets it.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from colosseum.core.errors import ConfigError

if TYPE_CHECKING:
    from colosseum.sp2.core.config import ColosseumConfig
    from colosseum.sp2.envs.game import GameSpec, MultiAgentEnv, RoleSpec
    from colosseum.sp2.networks.model import PolicyModel

_INJECTABLE = ("observation_space", "action_space", "global_state_space", "action_spec")


def import_class(dotted_path: str) -> type:
    """Import and return a class from ``"package.module.ClassName"``.

    Raises ValueError (no dot), ModuleNotFoundError, AttributeError, or TypeError (not a class).
    """
    if "." not in dotted_path:
        raise ValueError(f"dotted_path must be in the form 'module.ClassName', got: {dotted_path!r}")
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise TypeError(f"{dotted_path!r} resolved to {type(cls).__name__}, not a class")
    return cls


def _params(cls: type) -> set[str]:
    try:
        return set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return set()


def _inject(cls: type, role: RoleSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The role values ``cls.__init__`` names explicitly and ``kwargs`` does not already set."""
    from colosseum.sp2.core.specs import ActionSpec

    available = {
        "observation_space": role.observation_space,
        "action_space": role.action_space,
        "global_state_space": role.global_state_space,
        "action_spec": ActionSpec.from_space(role.action_space),
    }
    accepted = _params(cls)
    return {name: available[name] for name in _INJECTABLE if name in accepted and name not in kwargs}


def _construct(cls: type, role: RoleSpec, kwargs: dict[str, Any]) -> Any:
    return cls(**_inject(cls, role, kwargs), **kwargs)


def _with_in_dim(cls: type, kwargs: dict[str, Any], in_dim: int) -> dict[str, Any]:
    """``kwargs`` plus ``in_dim`` when ``cls.__init__`` names it (the framework's value wins)."""
    return {**kwargs, "in_dim": in_dim} if "in_dim" in _params(cls) else dict(kwargs)


def make_env(config: ColosseumConfig) -> MultiAgentEnv:
    """Instantiate ``config.env``; any failure becomes a ConfigError."""
    from colosseum.sp2.envs.game import MultiAgentEnv

    try:
        env = import_class(config.env.env_class)(**config.env.kwargs)
    except Exception as e:
        raise ConfigError(f"Failed to create env {config.env.env_class!r}: {type(e).__name__}: {e}") from e
    if not isinstance(env, MultiAgentEnv):
        raise ConfigError(f"env.env_class {config.env.env_class!r} must subclass "
                          f"colosseum.sp2.envs.game.MultiAgentEnv, got {type(env).__name__}")
    return env


def env_spec(config: ColosseumConfig) -> GameSpec:
    """The env's ``GameSpec``: built once from a fresh env, validated (EnvContractError), env closed."""
    from colosseum.sp2.envs.game import GameSpec

    env = make_env(config)
    try:
        spec = getattr(env, "spec", None)
        if not isinstance(spec, GameSpec):
            raise ConfigError(f"{config.env.env_class}.spec must be a colosseum.sp2.envs.game.GameSpec, "
                              f"got {type(spec).__name__}")
        spec.validate()
        return spec
    finally:
        env.close()


def build_model(agent_config: ColosseumConfig, role: RoleSpec) -> PolicyModel:
    """Build the agent's ``PolicyModel`` for ``role``'s spaces from ``agent_config.networks``.

    - ``model_class``: ``cls(**inject, **networks.kwargs)``; it must be a ``PolicyModel``.
    - otherwise a ``ComposedModel``: ``encoder(**inject, **kwargs)``;
      ``core(input_dim=encoder.latent_dim, **core.kwargs)`` (``NoCore`` when ``core`` is null);
      ``critic_encoder(**inject, **kwargs)`` if set (its role must declare a global state);
      ``policy(in_dim=core.output_dim, **inject, **kwargs)``;
      ``value(in_dim=core.output_dim + critic.output_dim, **kwargs)``. ``in_dim`` is passed only to
      heads that name it.
    """
    from colosseum.networks.cores import Core, NoCore
    from colosseum.sp2.networks.base import BaseCriticEncoder
    from colosseum.sp2.networks.composed import ComposedModel
    from colosseum.sp2.networks.model import PolicyModel

    net = agent_config.networks
    kwargs = dict(net.kwargs)
    if net.model_class:
        model_cls = import_class(net.model_class)
        if not issubclass(model_cls, PolicyModel):
            raise ConfigError(f"networks.model_class {net.model_class!r} must subclass "
                              f"colosseum.sp2.networks.model.PolicyModel, got {model_cls.__name__}")
        return _construct(model_cls, role, kwargs)

    encoder = _construct(import_class(net.encoder_class), role, kwargs)
    if net.core is None:
        core = NoCore(encoder.latent_dim)
    else:
        core_cls = import_class(net.core.class_path)
        if not issubclass(core_cls, Core):
            raise ConfigError(f"networks.core.class {net.core.class_path!r} must subclass "
                              f"colosseum.networks.cores.Core, got {core_cls.__name__}")
        core = core_cls(input_dim=encoder.latent_dim, **net.core.kwargs)
    critic = None
    critic_dim = 0
    if net.critic_encoder_class:
        if role.global_state_space is None:
            raise ConfigError(f"networks.critic_encoder_class is set ({net.critic_encoder_class!r}) but the agent's "
                              f"role declares no global_state_space; remove it or add a global state to the role")
        critic_cls = import_class(net.critic_encoder_class)
        if not issubclass(critic_cls, BaseCriticEncoder):
            raise ConfigError(f"networks.critic_encoder_class {net.critic_encoder_class!r} must subclass "
                              f"colosseum.sp2.networks.base.BaseCriticEncoder, got {critic_cls.__name__}")
        critic = _construct(critic_cls, role, kwargs)
        critic_dim = int(critic.output_dim)
    policy_cls = import_class(net.policy_class)
    policy = _construct(policy_cls, role, _with_in_dim(policy_cls, kwargs, core.output_dim))
    value_cls = import_class(net.value_class)
    value = value_cls(**_with_in_dim(value_cls, kwargs, core.output_dim + critic_dim))
    return ComposedModel(encoder, core, policy, value, critic)
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/unit/test_agent_roles.py tests/unit/test_build_model_v2.py -q`
Expected: `17 passed`.

- [ ] **Step 6: Run the full fast suite and ruff**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw`
Expected: every test passes and there is no warnings summary.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/sp2/core/roles.py \
  src/colosseum/sp2/core/registry.py \
  tests/unit/test_agent_roles.py \
  tests/unit/test_build_model_v2.py
git commit -m "feat(sp2): agent role resolution, build_model with role injection, make_env and env_spec"
git push origin sp2-game-model
```

---

## Contract notes

Additions to the interface contract (no contract name is changed or removed). Later parts may rely on them.

1. **`colosseum.core.ipc` queue helpers (T0.1).** `QueueReader`, `release_command_queues(queues, timeout) -> list`, `queue_depths(queues) -> dict[str, int]` are public in `colosseum.core.ipc` (the overview lists `core.ipc` as reused unchanged after T0.1). The SP2 launcher copy (T5.4) imports them; it must not copy the old private `_QueueReader` / `_release_command_queues` / `_queue_depths`, which no longer exist.
2. **Spec classes (T1.2, T1.3).**
   - `ObsSpec.is_dict` (False: a bare array with leaf path `()`), `ObsSpec.__eq__`/`__hash__` by signature.
   - `ActionSpec.group_mask(mask, group)` returns the part of a normalized mask tree for one group (None for box groups or no mask). `ActionSpec.__eq__`/`__hash__` by signature. `Units.__eq__`/`__hash__`/`__repr__` (value equality).
   - `ObsSpec.from_space` / `ActionSpec.from_space`: `TypeError` for an unsupported space type, `ValueError` for an unsupported shape or start. `GameSpec.validate` turns both into `EnvContractError`; `validate_config` (T6.2) should turn them into `ConfigError`.
   - `ObsSpec.check` checks structure and shapes only, not dtypes (buffers cast on write, spec block 1 "проверяется дёшево").
   - `normalize_mask` requires `bool` mask arrays (`int8` is an `EnvContractError`, per the contract's "dtype(bool)"). `docs/ENV_GUIDE.md` (T8.6) must say so.
3. **`GameSpec` (T1.4).** `DEFAULT_ROLE = "player"`. Seat lists are stored as tuples (`__post_init__`). Role and layout names must match `[A-Za-z0-9_][A-Za-z0-9_-]*` (the agent-id rule; they appear in metric keys and `--layout`). `teams`, `layout_size`, `role_of`, `outcome_kind` raise `EnvContractError` for an unknown layout or seat.
4. **`EpisodeTracker` (T1.5)**, for `MatchRunner` (T3.3) and `validate` (T6.2):
   - additions `seat_returns() -> list[float]`, `eliminated_step(seat) -> int | None`, `team_result() -> (team_rank, team_score)`;
   - the team result is resolved inside `on_step` of the final step, so a bad `Outcome` raises there;
   - `on_step` returns `{}` in the final step;
   - `on_step` before `on_reset` or after `episode_over` raises `RuntimeError` (a caller bug, not an env error);
   - two checks beyond the spec's list, both loud by the spec's principle: a reward must be a finite number, and `global_state` for a seat whose role declares no `global_state_space` is an error;
   - observations, masks and global states of LIVE waiting seats are ignored; for EMPTY seats they are errors;
   - a reset with an empty `acting` counts as the first idle step.
5. **Vector envs (T1.6).**
   - `VectorEnv(env_fn, num_envs, *, index_offset=0)`, `VectorEnv.envs`, `SubprocessVectorEnv.num_workers`.
   - An unknown layout in `reset` is a `ValueError` (a caller error; the tracker's `EnvContractError` stays for envs). A bad env index is an `IndexError`.
   - Exceptions from child processes are re-raised in the parent with the child traceback attached as a note.
6. **Config v2 (T1.7).** `get_agent_config(agent_id)` returns a config with `agents == {}` as in SP1, so roles are read with `config.agent_roles(agent_id)` on the top-level config. `scripts/bench_throughput.py::_make_config` keeps building the OLD config until T7.3.
7. **Distributions (T2.1, T2.2).**
   - Accessors: `CategoricalDist.logits/.mask/.num_categories`, `MultiCategoricalDist.logits/.mask/.nvec`, `DiagGaussianDist.mean/.log_std/.action_dim`, `TreeDist.spec/.parts`, `UnitsDist.group/.units/.U/.B/.params/.unit_mask/.action_mask`.
   - Helper functions in `colosseum.sp2.networks.dist.leaf`: `masked_log_softmax`, `categorical_*`, `gaussian_*`, `sample_categorical`.
   - `apply_mask(None)` returns the same object; `DiagGaussianDist.apply_mask(<tensor>)` raises `ValueError`.
   - `TreeDist` accepts any `Distribution` per group with the right decider count, so a custom per-group distribution plugs in. A fully custom action distribution (e.g. autoregressive) can also be returned by the policy head directly; it only has to implement the protocol.
   - `sample()`/`mode()` of `UnitsDist` put 0 for absent units and empty rows.
8. **Model (T2.3).**
   - `PolicyModel._check_unroll_args(num_slots, state0)` is available to subclasses.
   - `act()` takes `log_probs` from `dist.log_prob`, which equals `unit_log_probs.sum(-1)` for every built-in distribution.
   - `NormalizeObs` always outputs float32 and validates `source`; `RunningMeanStd` is public in `colosseum.sp2.networks.normalization`.
   - `heads.make_distribution` is the T2.1 function re-exported.
9. **`build_model` (T2.4).**
   - Injection never overrides a key present in `networks.kwargs`.
   - `in_dim` is set by the framework when a head names it, overriding a user value (SP1 semantics).
   - The value head gets no space injection (contract: `value(in_dim=..., **kwargs)`).
   - A critic encoder class that is not a `BaseCriticEncoder` is a `ConfigError`.
10. **Test kit (`tests/game_helpers.py`).**
    - Extra keyword arguments with defaults on the toy games: `EliminationFFA(length=10)`, `TeamDeadTeammateGame(length=6, dead_at=2)`, `UnitsGame(length=6)`, `AsymmetricGame(length=5)`, `CoopGame(length=5)`, `GlobalStateGame(length=5, truncate_at=None)`. `AsymmetricGame`'s layout is `"1v2"` (roles `hunter` with `Discrete(5)`, `prey` with `Discrete(3)`).
    - Additions: `ScriptedGame`, `TOY_GAMES`, `sample_legal_action`, `play_episode`, `make_core`, `flatten_tree`, `params_tree`, `GenericEncoder`, `GenericCriticEncoder`, `TreePolicyHead`, `GenericValue`.
    - `RandomPolicy(role)` takes a `RoleSpec`; box parts are N(0, 1), not uniform.
    - `pyproject.toml` lists `game_helpers` in ruff's `known-first-party`. Part 02 must add `game_harness` there when it creates `tests/contract/game_harness.py`.
    - `make_test_config` (T5.4) can build composed models from the dotted paths `game_helpers.GenericEncoder`, `game_helpers.TreePolicyHead`, `game_helpers.GenericValue` and `game_helpers.GenericCriticEncoder`: they take their spaces by injection and accept `hidden` from `networks.kwargs`.
11. **Dependencies.** T2.3 needs T1.4 and T1.5 besides T2.2, because it extends `tests/game_helpers.py`.
