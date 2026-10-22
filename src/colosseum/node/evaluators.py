"""The node agent's evaluation section (SP5 spec 5.3, 2.6, 7.1; T5.1): evaluator children and the lease protocol on
the session, over the pure bookkeeping of ``node.leases``. Main loop only (``NodeAgent`` calls it; it alone spawns).

- **How many.** ``Hello.slots.evaluators`` is the node's slot count; the hub says how many to run
  (``Welcome.evaluators``, ``Command(set_evaluators)``), capped at the slots. ``tick`` starts ``evaluator-<i>`` for
  every wanted ``i`` that has no process (a disabled slot stays down) and retires every ``i`` beyond the wanted
  number: its lease is reported ``abandoned`` at once, ``Stop`` goes on its control connection, the connection
  closes and the supervisor forgets it (P-PF4-9: neither a restart nor a death, no ``proc_died``, no
  ``worker_restarts``); a retired process still alive after the node's child grace is terminated, then killed.
- **Children.** An evaluator's control hello makes it idle (``NodeLeases.evaluator_up``); ``EvalDone`` /
  ``EvalFailed`` become block outcomes (a failure's text names the evaluator's log). An exit while the run goes on is
  a death: the lease's blocks are reported ``crash`` and the agent's generic restart rule applies (``WORKER_RESTARTS``,
  ``proc_died`` / ``slot_disabled``; A2 / T4.2). An exit while the node stops or the hub is away (dormant) is not
  restarted (the supervisor forgets it; the next ``Welcome`` starts the wanted ones again); its blocks are
  ``abandoned`` and wait unacked (a dormant node resends them after the reconnect).
- **Leases.** ``BlockLease`` -> ``NodeLease``; a lease that arrives when no evaluator is wanted is abandoned at
  once. Its players' eval blobs are fetched from the hub (``BlobRequest(kind="eval")``) once per key and machine in the
  node's transfer pool; nothing is fetched while the cache is full (the lease waits; a fetch that hit a full cache
  is tried again once space returns; any other failed fetch fails the lease's blocks ``missing_weights``). Ready
  leases go to idle evaluators (``EvalTask``); ``cancel_leases`` -> ``EvalCancel``; ``BlockAck`` -> forgotten.
- **Order on the session.** Every outcome is sent when it is made (the loop reads children before it ticks), so
  the ``BlockResult``s of a finished lease always precede the ``BlockLeaseRequest`` its freed evaluator triggers
  (``tick``); after a reconnect (``on_session_up``) the unacked outcomes are resent before any request. No request
  while the cache is full (T4.4) or the session is down. Asked slots that got no lease within ``REASK_SEC`` are asked
  again (the hub caps a node's held plus waiting leases at its slots, so a repeated ask never over-grants).
- **Cache.** Eval blobs no lease uses beyond the newest ``EVAL_BLOBS_KEPT`` leave the machine cache.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.leases import BlockOutcome, NodeLease, NodeLeases
from colosseum.node.procs import ChildSpec

logger = logging.getLogger(__name__)

__all__ = ["EVALUATOR_PREFIX", "REASK_SEC", "RETIRE_KILL_SEC", "EvaluatorHost", "NodeEvaluators", "is_eval_key",
           "outcome_message"]

EVALUATOR_PREFIX = "evaluator-"
REASK_SEC = 10.0                # asked slots without a lease for this long are asked again
RETIRE_KILL_SEC = 1.0           # a retired evaluator still alive after terminate is killed after this


class EvaluatorHost(Protocol):
    """What ``NodeEvaluators`` needs of the node agent (``colosseum.node.agent._EvaluatorHost``)."""

    supervisor: Any
    fetches: Any
    connected: bool
    cache_full: bool
    stopping: bool
    dormant: bool
    slots: int

    def current_run(self) -> Any: ...

    def send(self, message: pb.NodeMessage) -> bool: ...

    def send_child(self, name: str, message: Any) -> bool: ...

    def close_child(self, name: str) -> None: ...

    def disabled(self, name: str) -> bool: ...


def is_eval_key(key: Any) -> bool:
    """A cache key of an evaluation blob (``blob_key("eval", ...)``, A18)."""
    return getattr(key, "kind", None) == "blob" and str(getattr(key, "ident", "")).startswith("eval-")


def outcome_message(outcome: BlockOutcome) -> pb.NodeMessage:
    """The ``BlockResult`` of one block outcome (a result that cannot be encoded fails the block as ``crash``)."""
    from colosseum.net.messages import result_to_proto

    msg = pb.BlockResult(lease_id=outcome.lease_id, block_id=outcome.block_id, failed=outcome.failed,
                         error_kind=outcome.error_kind, error=outcome.error)
    if outcome.results is not None:
        try:
            msg.results.extend(result_to_proto(r) for r in outcome.results)
        except Exception as e:  # noqa: BLE001 - one malformed result never stops the node
            msg = pb.BlockResult(lease_id=outcome.lease_id, block_id=outcome.block_id, failed=True,
                                 error_kind="crash", error=f"a result could not be encoded ({type(e).__name__}: {e})")
    return pb.NodeMessage(block_result=msg)


def _index(name: str) -> int:
    return int(name.rsplit("-", 1)[1])


class NodeEvaluators:
    """The evaluation section of a node (module docstring)."""

    def __init__(self, host: EvaluatorHost, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._host = host
        self._clock = clock
        self.leases = NodeLeases()
        self._wanted = 0
        self._retiring: list[tuple[Any, float, bool]] = []    # (process, deadline, terminated)
        self._asked_at = 0.0
        self._warned: set[str] = set()

    # ---- demand ---------------------------------------------------------------------------------------------

    @property
    def wanted(self) -> int:
        return self._wanted

    def set_wanted(self, n: int) -> None:
        """``Welcome.evaluators`` / ``Command(set_evaluators)``: the processes start or retire in ``tick``."""
        wanted = max(0, min(int(n), int(self._host.slots)))
        if wanted != self._wanted:
            logger.info(f"node: evaluators wanted: {wanted} (slots {self._host.slots})")
        self._wanted = wanted

    # ---- children ---------------------------------------------------------------------------------------------

    def on_connected(self, name: str) -> None:
        self.leases.evaluator_up(_index(name))

    def on_message(self, name: str, message: Any) -> None:
        from colosseum.net.local import EvalDone, EvalFailed

        index = _index(name)
        if isinstance(message, EvalDone):
            self._report(self.leases.on_done(index, message.lease_id, message.results, message.cancelled))
        elif isinstance(message, EvalFailed):
            log_path = self._log_path(name)
            self._report(self.leases.on_failed(index, message.lease_id, message.kind,
                                               f"{message.message} ({message.where}; log {log_path})"))
        else:
            logger.warning(f"node: {name} sent an unexpected {type(message).__name__}; ignored")

    def on_exit(self, name: str, code: int | None, *, final: bool) -> bool:
        """An evaluator exited: its lease is released (``crash``, or ``abandoned`` when ``final``: the node stops or
        the hub is away). Returns True when the exit is handled here (``final``: no restart; the supervisor forgets
        the process, so the next wanted count starts it again); False: the agent's restart rule applies."""
        log_path = self._log_path(name)
        if final:
            self._report(self.leases.evaluator_down(_index(name), error_kind="abandoned",
                                                    error=f"{name} stopped with the node (exit {code})"),
                         send=not self._host.stopping)
            self._host.supervisor.forget(name)
            return True
        self._report(self.leases.evaluator_down(_index(name), error_kind="crash",
                                                error=f"{name} died (exit {code}); log {log_path}"))
        return False

    def _log_path(self, name: str) -> str:
        run = self._host.current_run()
        return "" if run is None else str(run.log_dir / f"{name}.log")

    # ---- hub messages ---------------------------------------------------------------------------------------

    def on_lease(self, lease: pb.BlockLease) -> None:
        from colosseum.net.messages import block_task_from_proto

        try:
            tasks = [block_task_from_proto(b, num_envs=int(lease.num_envs)) for b in lease.blocks]
        except Exception as e:  # noqa: BLE001 - a malformed lease fails its blocks (replayed by the hub)
            logger.error(f"node: lease {lease.lease_id[:12]} cannot be read ({type(e).__name__}: {e})")
            self._report([BlockOutcome(lease.lease_id, b.block_id, None, "crash", f"malformed lease: {e}")
                          for b in lease.blocks])
            return
        if not tasks:
            return
        node_lease = NodeLease(lease.lease_id, tasks[0].source, tasks, int(lease.num_envs), tasks[0].deterministic)
        self.leases.on_lease(node_lease, have=self._host.current_run().reader.has)
        if self._wanted == 0 or self._host.stopping:      # no evaluator will play it: back to the hub at once
            self._report(self.leases.abandon(lease.lease_id, "abandoned", "this node runs no evaluator now"))

    def on_ack(self, ack: pb.BlockAck) -> None:
        self.leases.on_ack(ack.lease_id, list(ack.block_ids))

    def on_cancel(self, lease_ids: list[str]) -> None:
        from colosseum.net.local import EvalCancel

        for index, lease_id in self.leases.on_cancel(lease_ids):
            self._host.send_child(f"{EVALUATOR_PREFIX}{index}", EvalCancel(lease_id))

    def on_fetch_done(self, done: Any) -> None:
        """A ``BlobFetched`` of an eval key (the fetcher's bookkeeping is done by the caller)."""
        run = self._host.current_run()
        if run is None or run.serial != done.serial:
            return
        if done.error is not None and done.full is not None:
            return                                   # a full cache: fetched again once space returns
        self._report(self.leases.on_fetched(done.key, done.error is None))

    def on_session_up(self) -> None:
        """A new session: the unacked outcomes go first (before any lease request: ``tick`` comes later)."""
        _held, resend = self.leases.on_reconnect()
        for outcome in resend:
            if not self._host.send(outcome_message(outcome)):
                break
        if resend:
            logger.info(f"node: resent {len(resend)} unacknowledged block results after the reconnect")

    def held(self) -> list[str]:
        return self.leases.held()

    # ---- every pass -------------------------------------------------------------------------------------------

    def tick(self, run: Any, now: float, *, spawn: bool) -> None:
        """Retired processes, the wanted processes (``spawn``: not while the hub is away), fetches, assignments,
        the lease request, the cache."""
        from colosseum.net.local import EvalTask

        self._reap(now)
        if spawn and not self._host.stopping:
            self._reconcile(run, now)
        fetches = self._host.fetches
        if not self._host.cache_full:
            for key in sorted(self.leases.missing(), key=lambda k: (k.agent_id, k.ident)):
                if not fetches.blob_in_flight(key):
                    fetches.fetch_blob(run, key)
        for index, lease in self.leases.assignments():
            self._host.send_child(f"{EVALUATOR_PREFIX}{index}", EvalTask(lease.lease_id, lease.source, lease.blocks,
                                                                         lease.num_envs, lease.deterministic))
        if self.leases.asked and now - self._asked_at >= REASK_SEC:
            self.leases.reset_asked()                # no answer: ask again (the hub caps at the slots)
        free = self.leases.request(connected=self._host.connected and not self._host.cache_full
                                   and not self._host.stopping)
        if free:
            if self._host.send(pb.NodeMessage(lease_request=pb.BlockLeaseRequest(free_slots=free))):
                self._asked_at = now
            else:
                self.leases.reset_asked()
        for key in self.leases.drop_candidates(busy=fetches.blob_in_flight):
            try:
                run.writer.drop(key)
            except OSError as e:
                logger.warning(f"node: cannot remove {key.ident} from the weight cache: {e}")

    def _reconcile(self, run: Any, now: float) -> None:
        from colosseum.node.evaluator_main import evaluator_target

        supervisor = self._host.supervisor
        for name in [n for n in supervisor.names() if n.startswith(EVALUATOR_PREFIX)]:
            if _index(name) >= self._wanted:
                self._retire(run, name, now)
        for index in range(self._wanted):
            name = f"{EVALUATOR_PREFIX}{index}"
            if supervisor.get(name) is not None or self._host.disabled(name):
                continue
            kwargs = dict(name=name, log_dir=str(run.log_dir), index=index, node=run.number,
                          node_address=run.node_address, authkey=run.authkey, cache_root=str(run.root),
                          cache_prefix=run.prefix, config=run.config)
            incarnation = supervisor.start(ChildSpec(name, "evaluator", evaluator_target, kwargs, daemon=True))
            logger.info(f"node: started {name} (incarnation {incarnation})")

    def _retire(self, run: Any, name: str, now: float) -> None:
        """Stop an evaluator the hub no longer wants (P-PF4-9: neither a restart nor a death)."""
        from colosseum.net.local import Stop

        self._report(self.leases.evaluator_down(_index(name), error_kind="abandoned",
                                                error=f"{name} was stopped (set_evaluators)"))
        supervisor = self._host.supervisor
        proc = supervisor.get(name)
        self._host.send_child(name, Stop("set_evaluators"))
        self._host.close_child(name)
        supervisor.forget(name)
        if proc is not None and proc.exitcode is None:
            self._retiring.append((proc, now + float(run.grace), False))
        logger.info(f"node: stopped {name} (set_evaluators)")

    def _reap(self, now: float) -> None:
        keep = []
        for proc, deadline, terminated in self._retiring:
            if proc.exitcode is not None:
                proc.join(0)
                continue
            if now >= deadline:
                if not terminated:
                    proc.terminate()
                    keep.append((proc, now + RETIRE_KILL_SEC, True))
                else:
                    proc.kill()
                    proc.join(RETIRE_KILL_SEC)
                continue
            keep.append((proc, deadline, terminated))
        self._retiring = keep

    def _report(self, outcomes: list[BlockOutcome], *, send: bool = True) -> None:
        """Send block outcomes now (they stay unacked in ``NodeLeases`` until the hub's ``BlockAck``)."""
        if not send or not self._host.connected:
            return
        for outcome in outcomes:
            if not self._host.send(outcome_message(outcome)):
                return

    # ---- the end ----------------------------------------------------------------------------------------------

    def close(self) -> None:
        """The node or its run ends: retired processes still alive are killed; the books start afresh."""
        for proc, _deadline, _terminated in self._retiring:
            if proc.exitcode is None:
                proc.kill()
            proc.join(RETIRE_KILL_SEC)
        self._retiring = []
        self.leases = NodeLeases()
        self._wanted = 0
