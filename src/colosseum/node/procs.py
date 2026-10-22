"""Child processes of a node (SP5 spec 1, 7.1): spawn, named incarnations, restart limits, bounded stop.

Every child is started with ``utils.process.start_process`` (SIGINT blocked until its ``run_child`` body ignores it)
and runs ``init_child_process`` (``watch_parent``, re-exported here: ``PR_SET_PDEATHSIG`` on Linux, a watchdog thread
elsewhere, SP5 spec 7.8: it dies with the node). The
death signal fires when the *thread* that started the child exits, so the node starts its children only from its
main-loop thread. The supervisor never shares a lock, queue or event with a child: it only starts, polls
(``exitcode``), terminates and kills them; everything else goes over the children's own connections
(``net.local``)."""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from colosseum.utils.process import start_process, watch_parent  # watch_parent: the contract's node.procs name

logger = logging.getLogger(__name__)

__all__ = ["LEARNER_RESTARTS", "WORKER_RESTARTS", "ChildSpec", "ChildSupervisor", "watch_parent"]

WORKER_RESTARTS = (3, 600.0)        # max restarts per window (s): workers and evaluators (spec 7.1)
LEARNER_RESTARTS = (3, 1800.0)      # learners (T4.2 uses it)
_POLL_SEC = 0.05


@dataclass(frozen=True)
class ChildSpec:
    name: str                                       # "worker-0", "learner-main", "evaluator-0"
    role: Literal["worker", "learner", "evaluator"]
    target: Callable
    kwargs: dict = field(default_factory=dict)
    daemon: bool = True                             # False for workers with a subprocess vector env (P-PF2-7)


@dataclass
class _Child:
    spec: ChildSpec
    proc: Any
    incarnation: int
    reported: bool = False                          # exit returned by exited() already


class ChildSupervisor:
    """Children by name. ``start`` gives a new incarnation (1, 2, ...) per name and passes it to the target as the
    ``incarnation`` keyword; ``exited`` returns the children that ended since its last call; ``may_restart`` applies
    the restart limit of the child's role in a sliding window (``clock``: monotonic); ``stop_all(grace)`` waits up to
    ``grace`` for every child to exit (the caller sent ``Stop`` first), then terminates and kills the rest (about
    1 s each) and returns their names."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic, ctx: Any = None) -> None:
        self._clock = clock
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._children: dict[str, _Child] = {}
        self._incarnations: dict[str, int] = {}
        self._restarts: dict[str, deque[float]] = {}

    def start(self, spec: ChildSpec) -> int:
        incarnation = self._incarnations.get(spec.name, 0) + 1
        self._incarnations[spec.name] = incarnation
        proc = self._ctx.Process(target=spec.target, name=spec.name, daemon=spec.daemon,
                                 kwargs={**spec.kwargs, "incarnation": incarnation})
        start_process(proc)
        self._children[spec.name] = _Child(spec, proc, incarnation)
        return incarnation

    def get(self, name: str) -> Any | None:
        child = self._children.get(name)
        return None if child is None else child.proc

    def role(self, name: str) -> str | None:
        child = self._children.get(name)
        return None if child is None else child.spec.role

    def incarnation(self, name: str) -> int:
        return self._incarnations.get(name, 0)

    def names(self) -> list[str]:
        return list(self._children)

    def alive(self) -> list[str]:
        """Names of the children whose process has not exited."""
        return [n for n, c in self._children.items() if c.proc.exitcode is None]

    def sentinels(self) -> dict[int, str]:
        """Process sentinels of the live children (for ``multiprocessing.connection.wait``)."""
        return {c.proc.sentinel: n for n, c in self._children.items() if c.proc.exitcode is None}

    def exited(self) -> list[tuple[str, int | None]]:
        out = []
        for name, child in self._children.items():
            if not child.reported and child.proc.exitcode is not None:
                child.reported = True
                out.append((name, child.proc.exitcode))
        return out

    def may_restart(self, name: str, now: float | None = None) -> bool:
        """Whether ``name`` may start once more: fewer than the role's limit of restarts in the window (and the
        restart is then recorded)."""
        child = self._children.get(name)
        if child is None:
            return False
        limit, window = LEARNER_RESTARTS if child.spec.role == "learner" else WORKER_RESTARTS
        now = self._clock() if now is None else now
        times = self._restarts.setdefault(name, deque())
        while times and now - times[0] > window:
            times.popleft()
        if len(times) >= limit:
            return False
        times.append(now)
        return True

    def restart(self, name: str) -> int:
        return self.start(self._children[name].spec)

    def forget(self, name: str) -> None:
        self._children.pop(name, None)

    def stop_all(self, grace: float) -> list[str]:
        deadline = time.monotonic() + max(0.0, grace)
        while time.monotonic() < deadline and any(c.proc.is_alive() for c in self._children.values()):
            time.sleep(_POLL_SEC)
        remaining = [n for n, c in self._children.items() if c.proc.is_alive()]
        for name in remaining:
            self._children[name].proc.terminate()
        self._join(remaining, 1.0)
        for name in [n for n in remaining if self._children[n].proc.is_alive()]:
            self._children[name].proc.kill()
        self._join(remaining, 1.0)
        if remaining:
            logger.warning(f"node: terminated children that did not stop within {grace:.1f} s: {remaining}")
        return remaining

    def _join(self, names: list[str], timeout: float) -> None:
        end = time.monotonic() + timeout
        for name in names:
            self._children[name].proc.join(max(0.0, end - time.monotonic()))
