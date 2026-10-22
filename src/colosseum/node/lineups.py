"""Lineups and the weight cache of a node (SP5 spec 4.2, 4.6; ruling R11; controller ruling R31): grants that wait
until every weight they name is in the machine's cache, the fetches they start (snapshots and ``Welcome.fixed`` blobs
from the hub, other machines' latest weights through ``node.fetch``), the failed-fetch path (the old lineup stays,
``lineup_fetch_failed``, the cell asks again) and the removal of snapshots no lineup names any more.

Owned by the node agent's main loop (``colosseum.node.agent``); ``NodeLineups`` talks to the agent only through
``LineupHost`` (its ``_LineupHost`` adapter): the cell tracker, the current run, whether the node stops or is
connected, sending to the hub, a counter, the agents whose learner runs here, and sending to a worker.

While the cache is full (``LineupHost.cache_full``, SP5 T4.4, spec 7.1) the node takes no new work: no lineup
requests and no fetches (the cells keep playing the lineups they have; grants already waiting stay until their
weights arrive or they are given up).

The lineup-after-cache invariant (``prune_cache``): a cache key leaves only after the lineup replacing it was sent to
the worker, and workers pump their control connection before every cache read.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.books import CellTracker
from colosseum.node.fetch import BlobFetched, LatestFetched, LatestFetchFailed, NodeFetcher

logger = logging.getLogger(__name__)

__all__ = ["CACHE_DROP_DELAY_SEC", "FIXED_RETRY_SEC", "LINEUP_WAIT_SEC", "BadSeat", "LineupHost", "NodeLineups"]

LINEUP_WAIT_SEC = 30.0          # a grant waiting longer than max(ttl, this) for its weights is given up (P-PF2-12)
CACHE_DROP_DELAY_SEC = 10.0     # an unreferenced snapshot stays this long in the cache before it is removed
FIXED_RETRY_SEC = 2.0           # a failed fetch of a Welcome.fixed blob is retried after this


class BadSeat(ValueError):
    """A grant names a player this node cannot seat; ``player`` = (agent, network) (one WARNING per player)."""

    def __init__(self, message: str, agent: str, network: str) -> None:
        super().__init__(message)
        self.player = (agent, network)


class LineupHost(Protocol):
    """What ``NodeLineups`` uses of the node agent."""

    @property
    def cells(self) -> CellTracker: ...
    @property
    def stopping(self) -> bool: ...
    @property
    def connected(self) -> bool: ...
    @property
    def cache_full(self) -> bool: ...                 # no new work while set (T4.4)
    def current_run(self) -> Any: ...
    def send(self, message: pb.NodeMessage) -> bool: ...
    def count(self, name: str, n: int = 1) -> None: ...
    def learner_agents(self) -> list[str]: ...
    def worker_connected(self, worker: int) -> bool: ...
    def send_worker(self, worker: int, message: Any) -> bool: ...


class NodeLineups:
    """The node's lineups and cache bookkeeping (module docstring). ``cached``: snapshot keys this node put into the
    cache (pruned when unreferenced)."""

    def __init__(self, host: LineupHost, fetches: NodeFetcher) -> None:
        self._host = host
        self.fetches = fetches
        self._fetch_warned: set = set()
        self._refused_players: set = set()        # (agent, network) of refused grants (warned once)
        self.cached: set = set()                    # snapshot keys this node put into the cache
        self._unref_since: dict[Any, float] = {}

    def clear(self) -> None:
        """The run was released: its cache keys are forgotten (the files went with the run)."""
        self.cached.clear()
        self._unref_since.clear()

    def needed_fixed(self, run: Any) -> list:
        """The ``Welcome.fixed`` blobs this node needs: frozen and teacher weights always, an agent's ``init`` weights
        only while its learner runs (or is about to run) here (spec 4.6: "или назначенному лёрнеру")."""
        learners = set(self._host.learner_agents())
        return [k for k, kind in run.fixed_keys.items() if kind != "init" or k.agent_id in learners]

    def fetch_fixed(self, run: Any, now: float) -> None:
        """Fetch the needed ``Welcome.fixed`` blobs that are not cached (a failed one again after
        ``FIXED_RETRY_SEC``; none while the cache is full)."""
        if self._host.cache_full:
            return
        for key in self.needed_fixed(run):
            if key in run.fixed_done or self.fetches.blob_in_flight(key) or now < run.fixed_retry_at.get(key, 0.0):
                continue
            if run.reader.has(key):
                run.fixed_done.add(key)
                continue
            self.fetches.fetch_blob(run, key)

    def on_grants(self, grants: pb.LineupGrants) -> None:
        from colosseum.net.messages import lineup_from_proto

        run = self._host.current_run()
        if run is None or self._host.stopping:
            return
        for grant in grants.grants:
            try:
                lineup = lineup_from_proto(grant.lineup)
                lineup.epoch = int(grant.epoch)
                keys = self.seat_keys(run, lineup)
            except ValueError as e:                  # the cell asks again after LINEUP_RETRY_SEC: warn once
                player = getattr(e, "player", None) or ("", str(e))
                if player not in self._refused_players:
                    self._refused_players.add(player)
                    logger.warning(f"node: refused a lineup grant for worker {grant.worker}, env {grant.env}: {e} "
                                   f"(further grants naming it are refused silently)")
                continue
            self._host.cells.granted(int(grant.worker), int(grant.env), int(grant.epoch), float(grant.ttl_sec),
                                     lineup, keys)
        self.deliver(run)

    def seat_keys(self, run: Any, lineup: Any) -> dict:
        """The cache key of every seat's weights (none for a scripted seat); ValueError for an unknown player."""
        from colosseum.arena.entities import snapshot_version
        from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID
        from colosseum.node.weights import CacheKey, segment_name

        keys = {}
        for i, seat in enumerate(lineup.seats):
            agent, net = seat.agent_id, seat.network_id
            if net == FIXED_NETWORK_ID:
                if agent in run.frozen_keys:
                    keys[i] = run.frozen_keys[agent]
                elif agent not in run.bots:
                    raise BadSeat(f"seat {i}: {agent!r} is no fixed player of this run", agent, net)
            elif agent not in run.agent_ids:
                raise BadSeat(f"seat {i}: {agent!r} is no trainable agent of this run", agent, net)
            elif net == LATEST_NETWORK_ID:
                keys[i] = CacheKey("latest", agent)
            elif snapshot_version(net) is not None:
                key = CacheKey("snapshot", agent, net)
                try:
                    segment_name(run.prefix, key)
                except ValueError as e:
                    raise BadSeat(f"seat {i}: {e}", agent, net) from None
                keys[i] = key
            else:
                raise BadSeat(f"seat {i}: unknown network {net!r}", agent, net)
        return keys

    def has(self, run: Any, key: Any) -> bool:
        if key.kind == "latest":
            return run.reader.latest_version(key.agent_id) is not None
        return key in run.fixed_done or key in self.cached or run.reader.has(key)

    def request_lineups(self) -> None:
        """One ``LineupRequest`` for every due cell; none while the hub is away (expired cells keep their lineups and
        are all requested in the first pass after the ``Welcome``) or the cache is full."""
        if not self._host.connected or self._host.cache_full:
            return
        due = self._host.cells.due()
        if due:
            self._host.send(pb.NodeMessage(lineup_request=pb.LineupRequest(cells=[
                pb.CellRequest(worker=w, env=e, epoch=ep, incarnation=i) for w, e, ep, i in due])))

    def expire_waiting(self) -> None:
        for worker, env, keys in self._host.cells.stale_waiting(LINEUP_WAIT_SEC):
            self._host.cells.failed(worker, env)
            self._host.count("lineup_fetch_failed")
            run = self._host.current_run()
            missing = [k for k in keys.values() if run is None or not self.has(run, k)]
            for key in missing:
                self.warn_fetch(key, f"its weights did not arrive within {LINEUP_WAIT_SEC:.0f} s")

    def deliver(self, run: Any) -> None:
        """Send every waiting grant whose weights are all cached to its worker (R11); fetch what is missing."""
        from colosseum.net.local import CellLineup, WorkerLineups

        ready: dict[int, list] = {}
        for worker, env, _lineup, keys in self._host.cells.waiting():
            missing = [k for k in set(keys.values()) if not self.has(run, k)]
            for key in missing:
                if key.kind != "latest" and not self.fetches.blob_in_flight(key) and not self._host.cache_full:
                    self.fetches.fetch_blob(run, key)     # latest weights: pump_fetches
            if missing or not self._host.worker_connected(worker):
                continue
            lineup, keys = self._host.cells.deliver(worker, env)
            ready.setdefault(worker, []).append(CellLineup(env, lineup, dict(keys)))
        for worker, cells in ready.items():
            self._host.send_worker(worker, WorkerLineups(cells))

    def on_fetch_done(self, done: BlobFetched) -> None:
        self.fetches.blob_done(done)
        run = self._host.current_run()
        if run is None or run.serial != done.serial:
            return
        if done.error is None:
            if done.key in run.fixed_keys:
                run.fixed_done.add(done.key)
            else:
                self.cached.add(done.key)
            self._fetch_warned.discard(done.key)
            if not self._host.stopping:
                self.deliver(run)
            return
        if done.key in run.fixed_keys:
            run.fixed_retry_at[done.key] = time.monotonic() + FIXED_RETRY_SEC
            self.warn_fetch(done.key, f"{done.error}; retrying (the workers start once every fixed blob is here)")
            return
        failed = [(w, e) for w, e, _lineup, keys in self._host.cells.waiting() if done.key in keys.values()]
        for worker, env in failed:
            self._host.cells.failed(worker, env)
            self._host.count("lineup_fetch_failed")
        self.warn_fetch(done.key, f"{done.error}; {len(failed)} lineups keep their old one")

    def pump_fetches(self, run: Any, now: float) -> None:
        """Node item 11: the latest transfers the planner wants now (the agents named by latest seats of this node's
        live and waiting lineups are refreshed; never the agents whose learner runs here; nothing while the cache is
        full)."""
        if self._host.cache_full:
            return
        referenced = {k.agent_id for k in self._host.cells.referenced() if k.kind == "latest"}
        self.fetches.pump(run, now, referenced=referenced, local=self._host.learner_agents(),
                          hub_online=self._host.connected)

    def on_latest_fetched(self, ev: LatestFetched) -> None:
        run = self._host.current_run()
        if run is None or run.serial != ev.serial or run.closed:
            return
        outcome = self.fetches.latest_done(run, ev)
        if outcome == "published":
            if not self._host.stopping:
                self.deliver(run)
        elif outcome == "failed":
            self.fail_latest_grants(run, ev.agent_id)

    def on_latest_fetch_failed(self, ev: LatestFetchFailed) -> None:
        run = self._host.current_run()
        if run is None or run.serial != ev.serial:
            return
        self.fetches.latest_failed(ev)
        self.fail_latest_grants(run, ev.agent_id)

    def fail_latest_grants(self, run: Any, agent: str) -> None:
        """A latest transfer of ``agent`` failed and nothing of it is cached: the grants waiting for it keep their old
        lineups, count ``lineup_fetch_failed`` and are requested again (spec 4.2; R11)."""
        from colosseum.node.weights import CacheKey

        key = CacheKey("latest", agent)
        if self.has(run, key):
            return
        for worker, env, _lineup, keys in self._host.cells.waiting():
            if key in keys.values():
                self._host.cells.failed(worker, env)
                self._host.count("lineup_fetch_failed")

    def warn_fetch(self, key: Any, why: str) -> None:
        if key in self._fetch_warned:
            return
        self._fetch_warned.add(key)
        logger.warning(f"node: fetching {key.kind} {key.agent_id}/{key.ident} failed: {why} (lineup_fetch_failed; "
                       f"further failures of this key are only counted)")

    def prune_cache(self, run: Any, now: float) -> None:
        """Snapshots no cell references for ``CACHE_DROP_DELAY_SEC`` leave the cache (``Welcome.fixed`` blobs and
        latest pointers stay).

        Invariant (a worker never misses a delivered snapshot): a key stops being referenced only when ``deliver``
        replaces its cell's lineup, and ``deliver`` sends the replacing ``WorkerLineups`` on the same ordered control
        connection at once; a worker pumps every pending control message before each cache read
        (``WorkerIO.poll_command``), so it never reads a lineup that was replaced, and the snapshot of the lineup it
        does read is still referenced. Any new delivery path must keep this (replace a cell's keys only together
        with sending the replacing lineup). The delay is a margin, not what makes it safe."""
        referenced = self._host.cells.referenced()
        for key in list(self.cached):
            if key in referenced or self.fetches.blob_in_flight(key):
                self._unref_since.pop(key, None)
                continue
            since = self._unref_since.setdefault(key, now)
            if now - since >= CACHE_DROP_DELAY_SEC:
                try:
                    run.writer.drop(key)
                except OSError as e:
                    logger.warning(f"node: cannot remove {key} from the weight cache: {e}")
                self.cached.discard(key)
                self._unref_since.pop(key, None)
