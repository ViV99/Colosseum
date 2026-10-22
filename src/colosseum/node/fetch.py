"""A node's downloads (SP5 spec 4.5, 4.6; amendment A25; criterion 4): which latest weights it fetches from other
machines and when (``FetchPlanner``, pure, node clock), and the transfers themselves (``NodeFetcher``: hub blobs and
latest weights, in the node's transfer pool).

Weights reach a node's workers only through the machine's shared-memory cache (R10). For an agent whose learner
runs on THIS node the learner writes the cache itself; every other agent's latest weights are fetched by the node,
once per (agent, version), and published into the cache, so the number of network downloads does not depend on
the number of workers (criterion 4). Snapshots and ``Welcome.fixed`` blobs (frozen, teacher, ``init``) come from the
hub by id (``Download``).

Rules (``FetchPlanner.due``):
- the planner knows, per agent, the newest ``VersionInfo`` it heard of (``Welcome.versions``, ``Announce.versions``;
  a re-announce of the same version with new addresses replaces the source);
- agents whose learner runs on this node (``set_local_agents``) are never fetched;
- at most one transfer per agent is in flight;
- with the hub online: an agent is fetched when its announced version is newer than the cached one and either
  nothing is cached yet (first weights: at once) or a latest seat of a live or pending lineup of this node names it
  (``set_referenced``); the version check runs at most once per ``rollout.weight_sync_interval_sec`` per agent
  (the cadence starts when a transfer starts, so a failure also waits one interval);
- without the hub (session down): the node asks the learner's data port directly (``GetLatest(agent,
  since_version)``) once per interval for every referenced or never-cached agent whose source has data addresses;
  a learner reachable only through the relay gets no polls (spec 4.5: weights of such a learner need the hub);
- a result never lowers the cached version (versions only grow, R9): a version at or below the cached one is
  recorded as redundant (counted; criterion 4 expects zero).

``NodeFetcher`` runs the transfers in the node's transfer pool; its threads touch no main-loop state: they post
``BlobFetched`` / ``LatestFetched`` / ``LatestFetchFailed`` into the node inbox, and the main loop decides what the
result means for the cache and the lineups (``colosseum.node.agent``: a lineup reaches a worker only once all its
weights are cached, R11; a failed fetch keeps the old lineup and counts ``lineup_fetch_failed``).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import grpc

from colosseum.net.proto import colosseum_v1_pb2 as pb

logger = logging.getLogger(__name__)

__all__ = [
    "DOWNLOAD_TIMEOUT_SEC", "BlobFetched", "FetchJob", "FetchPlanner", "FetchStats", "LatestFetchFailed",
    "LatestFetched", "NodeFetcher",
]

DOWNLOAD_TIMEOUT_SEC = 300.0


class VersionLike(Protocol):        # colosseum.hub.state.VersionInfo
    agent_id: str
    generation: int
    version: int
    node: int
    addresses: tuple[str, ...]
    relay_only: bool


# ---------------------------------------------------------------------------------------------------------------
# The planner (pure)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchJob:
    """One transfer to start: ``mode`` "announced" (the hub said ``source.version`` exists) or "poll" (no hub:
    ``GetLatest`` answers "not modified" or with whatever is newer than ``since_version``)."""

    agent_id: str
    since_version: int                 # the cached version, -1 when nothing is cached
    source: VersionLike
    mode: Literal["announced", "poll"]


@dataclass
class FetchStats:
    jobs: int = 0                      # transfers started
    fetched: int = 0                   # transfers that brought a newer version (one network download each)
    not_modified: int = 0              # polls answered "not modified"
    failed: int = 0
    redundant: int = 0                 # a version at or below the cached one arrived (criterion 4: 0)
    history: dict[str, list[int]] = field(default_factory=dict)   # versions fetched per agent, in order

    def as_counters(self) -> dict[str, int]:
        """Cumulative values for ``Heartbeat.counters`` (names of the hub's CounterBook, A2)."""
        return {"weight_fetches": self.fetched, "weight_fetch_failed": self.failed,
                "weight_fetch_not_modified": self.not_modified, "weight_fetch_redundant": self.redundant}


class FetchPlanner:
    """Fetch decisions of one node for one run (module docstring). Not thread-safe: the node main loop owns it.
    ``stats`` may be shared by the planners of successive runs of one node process (its counters are cumulative per
    process: ``Heartbeat.counters``)."""

    def __init__(self, *, node: int, sync_interval_sec: float, stats: FetchStats | None = None) -> None:
        if sync_interval_sec <= 0:
            raise ValueError(f"sync_interval_sec must be > 0, got {sync_interval_sec}")
        self._node = int(node)
        self._interval = float(sync_interval_sec)
        self._known: dict[str, VersionLike] = {}
        self._local: set[str] = set()
        self._referenced: set[str] = set()
        self._in_flight: dict[str, FetchJob] = {}
        self._next_try: dict[str, float] = {}
        self.stats = stats if stats is not None else FetchStats()

    # --- inputs ------------------------------------------------------------------------------------------------

    def on_version(self, info: VersionLike) -> bool:
        """Remember ``info`` if it is newer than what is known for its agent, or the same version from a new place
        (new addresses after the learner node rebound its data port). Returns True when it was taken."""
        known = self._known.get(info.agent_id)
        if known is not None:
            if info.version < known.version:
                return False
            if info.version == known.version and (info.node, tuple(info.addresses), info.relay_only) == (
                    known.node, tuple(known.addresses), known.relay_only):
                return False
        self._known[info.agent_id] = info
        return True

    def set_local_agents(self, agents: Iterable[str]) -> None:
        self._local = set(agents)

    def set_referenced(self, agents: Iterable[str]) -> None:
        self._referenced = set(agents)

    def forget(self, agent_id: str) -> None:
        """The agent left the run (``AgentState.status`` ``removed``)."""
        self._known.pop(agent_id, None)
        self._next_try.pop(agent_id, None)

    # --- decisions ---------------------------------------------------------------------------------------------

    @property
    def interval(self) -> float:
        return self._interval

    def known(self, agent_id: str) -> VersionLike | None:
        return self._known.get(agent_id)

    def in_flight(self, agent_id: str) -> bool:
        return agent_id in self._in_flight

    def due(self, cached: Mapping[str, int], *, hub_online: bool, now: float) -> list[FetchJob]:
        """Transfers to start now; each returned job is marked in flight (``finished`` / ``failed`` end it)."""
        jobs: list[FetchJob] = []
        for agent_id in sorted(self._known):
            info = self._known[agent_id]
            if agent_id in self._local or info.node == self._node or agent_id in self._in_flight:
                continue
            if now < self._next_try.get(agent_id, -math.inf):
                continue
            have = int(cached.get(agent_id, -1))
            wanted = have < 0 or agent_id in self._referenced
            if not wanted:
                continue
            if hub_online:
                if info.version <= have:
                    continue
                job = FetchJob(agent_id, have, info, "announced")
            else:
                if info.relay_only or not info.addresses:
                    continue
                job = FetchJob(agent_id, have, info, "poll")
            self._in_flight[agent_id] = job
            self._next_try[agent_id] = now + self._interval
            self.stats.jobs += 1
            jobs.append(job)
        return jobs

    def finished(self, agent_id: str, version: int | None, *, cached: int) -> bool:
        """The transfer of ``agent_id`` ended: ``version`` arrived (None: "not modified"); ``cached`` is the cache's
        version before publishing. Returns True when the caller should publish (a newer version arrived)."""
        self._in_flight.pop(agent_id, None)
        if version is None:
            self.stats.not_modified += 1
            return False
        if version <= cached:
            self.stats.redundant += 1
            return False
        self.stats.fetched += 1
        self.stats.history.setdefault(agent_id, []).append(int(version))
        return True

    def failed(self, agent_id: str) -> None:
        """The transfer failed (unreachable, refused, bad digest, cache full): retried after one interval."""
        self._in_flight.pop(agent_id, None)
        self.stats.failed += 1


# ---------------------------------------------------------------------------------------------------------------
# Transfers (the node's transfer pool)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class BlobFetched:
    """A hub blob (snapshot or ``Welcome.fixed``) is in the cache (``error`` None) or could not be fetched."""

    key: Any
    error: str | None
    serial: int                        # the run this fetch belongs to (a fetch of an older run is ignored)


@dataclass
class LatestFetched:
    """A latest-weights transfer ended: ``fetched`` the checked frame (``dataport.FetchedLatest``), None for "not
    modified"."""

    agent_id: str
    fetched: Any
    serial: int


@dataclass
class LatestFetchFailed:
    agent_id: str
    reason: str                        # FetchError.reason, "unreachable", "cancelled", ...
    detail: str
    serial: int


@dataclass
class _DataLink:
    """How this node opens channels to other nodes' data ports (set per run)."""

    make_channel: Callable[[str], grpc.Channel] | None    # None: this node cannot (no certificate)
    metadata: tuple[tuple[str, str], ...]
    connect_timeout_sec: float
    max_message_mb: int
    why_not: str = ""


class NodeFetcher:
    """The downloads of a node (module docstring). The main loop calls everything except the ``_*_job`` bodies,
    which run in the transfer pool (``submit``) and report through ``post``. ``stats`` (the planner's counters) and
    ``fetch_bytes`` are cumulative for the node process."""

    def __init__(self, *, submit: Callable[..., Future], post: Callable[[object], None]) -> None:
        self._submit = submit
        self._post = post
        self.stats = FetchStats()
        self.fetch_bytes = 0
        self.planner: FetchPlanner | None = None
        self._hub_channel: grpc.Channel | None = None
        self._hub_metadata: tuple[tuple[str, str], ...] = ()
        self._blobs: dict[Any, Future] = {}             # cache key -> its download
        self._latest: dict[str, Future] = {}            # agent -> its latest transfer
        self._link: _DataLink | None = None
        self._channels_lock = threading.Lock()
        self._channels: dict[tuple, tuple[str, grpc.Channel]] = {}   # (node, addresses) -> (address, channel)
        self._warned: set[tuple[str, str]] = set()      # (agent, reason) of latest failures warned in this run

    # ---- setup ----------------------------------------------------------------------------------------------

    def set_hub(self, channel: grpc.Channel, metadata: Sequence[tuple[str, str]]) -> None:
        self._hub_channel, self._hub_metadata = channel, tuple(metadata)

    def start_run(self, *, number: int, sync_interval_sec: float, link: _DataLink) -> None:
        """A new run (first ``Welcome``): a planner of this node number; the data-port link."""
        self.planner = FetchPlanner(node=number, sync_interval_sec=sync_interval_sec, stats=self.stats)
        self._link = link
        self._warned.clear()

    def clear_run(self) -> None:
        """The run was released: its transfers are forgotten (they were cancelled and waited for)."""
        self._blobs.clear()
        self._latest.clear()
        self.planner = None
        self.close_channels()

    def futures(self) -> list[Future]:
        return [*self._blobs.values(), *self._latest.values()]

    def close_channels(self) -> None:
        with self._channels_lock:
            channels, self._channels = list(self._channels.values()), {}
        for _address, channel in channels:
            channel.close()

    @staticmethod
    def link(*, insecure: bool, token_secret: str, identity: Any, max_message_mb: int,
             connect_timeout_sec: float) -> _DataLink:
        """The data-port link of a node: plain + the token secret (``--insecure``), else mutual TLS with the node's
        certificate (a node without one, ``train``'s local node, cannot reach other data ports)."""
        from colosseum.net.grpc_util import channel_options
        from colosseum.net.security import MD_TOKEN, NODE_SERVER_NAME, data_channel_credentials

        # Every data channel has its own subchannels: a probe after a failed one connects at once instead of waiting
        # for the reconnect backoff of a subchannel shared through grpc's global pool.
        own_pool = [("grpc.use_local_subchannel_pool", 1)]
        if insecure:
            def plain(address: str) -> grpc.Channel:
                return grpc.insecure_channel(address, options=channel_options(max_message_mb) + own_pool)
            return _DataLink(plain, ((MD_TOKEN, token_secret),), float(connect_timeout_sec), int(max_message_mb))
        if identity is None or identity.cert_pem is None:
            return _DataLink(None, (), float(connect_timeout_sec), int(max_message_mb),
                             why_not="this node has no node certificate for data ports")
        credentials = data_channel_credentials(identity, (identity.path / "ca.crt").read_bytes())

        def tls(address: str) -> grpc.Channel:
            return grpc.secure_channel(address, credentials,
                                       options=channel_options(max_message_mb, server_name=NODE_SERVER_NAME)
                                       + own_pool)
        return _DataLink(tls, (), float(connect_timeout_sec), int(max_message_mb))

    # ---- hub blobs ------------------------------------------------------------------------------------------

    def blob_in_flight(self, key: Any) -> bool:
        return key in self._blobs

    def fetch_blob(self, run: Any, key: Any) -> None:
        """Download ``key`` (a snapshot or a ``Welcome.fixed`` blob) from the hub into the cache; ``BlobFetched``
        follows."""
        future = self._submit(self._blob_job, run, key)
        self._blobs[key] = future
        serial = run.serial

        def done(f: Future) -> None:
            error = None
            if f.cancelled():
                error = "cancelled"
            elif f.exception() is not None:
                error = rpc_text(f.exception())
            self._post(BlobFetched(key, error, serial))

        future.add_done_callback(done)

    def blob_done(self, done: BlobFetched) -> None:
        self._blobs.pop(done.key, None)

    def download_bytes(self, run: Any, kind: str, agent_id: str, ident: str) -> bytes:
        """A transfer-pool thread: one hub ``Download`` (checked: size, offsets, digest; capped)."""
        from colosseum.net.frame import payload_byte_cap
        from colosseum.net.grpc_util import receive_parts
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

        parts = pb_grpc.HubStub(self._hub_channel).Download(
            pb.BlobRequest(kind=kind, agent_id=agent_id, ident=ident), metadata=self._hub_metadata,
            timeout=DOWNLOAD_TIMEOUT_SEC)
        data, _meta = receive_parts(parts, cap=payload_byte_cap(run.config.distributed.max_message_mb))
        if data is None:
            raise ValueError("the hub answered not_modified")
        return data

    def _blob_job(self, run: Any, key: Any) -> None:
        """A transfer-pool thread: ``Download`` the key's frame, check it, put it into the cache."""
        from colosseum.net.chunks import decode_weights
        from colosseum.node.weights import split_blob_key

        if key.kind == "snapshot":
            kind, ident = "snapshot", key.ident
        else:
            kind, ident = split_blob_key(key)
        data = self.download_bytes(run, kind, key.agent_id, ident)
        _state, meta = decode_weights(data, cap=len(data))
        if meta.agent_id != key.agent_id:
            raise ValueError(f"the frame holds weights of {meta.agent_id!r}")
        if run.closed:
            raise RuntimeError("the run was torn down (stopped or released) meanwhile")
        run.writer.put(key, data)

    # ---- latest weights of other machines' learners ----------------------------------------------------------

    def pump(self, run: Any, now: float, *, referenced: Iterable[str], local: Iterable[str],
             hub_online: bool) -> None:
        """Start the latest transfers the planner wants now (``_latest_job`` in the pool)."""
        planner = self.planner
        if planner is None:
            return
        planner.set_local_agents(local)
        planner.set_referenced(referenced)
        cached: dict[str, int] = {}
        for agent in run.agent_ids:
            have = run.reader.latest_version(agent)
            if have is not None:
                cached[agent] = int(have[0])
        for job in planner.due(cached, hub_online=hub_online, now=now):
            future = self._submit(self._latest_job, job, run.serial)
            self._latest[job.agent_id] = future
            serial, agent = run.serial, job.agent_id

            def done(f: Future, agent: str = agent, serial: int = serial) -> None:
                if f.cancelled():
                    self._post(LatestFetchFailed(agent, "cancelled", "", serial))
                elif f.exception() is not None:   # _latest_job reports its own failures; this is a bug in it
                    self._post(LatestFetchFailed(agent, "error", rpc_text(f.exception()), serial))

            future.add_done_callback(done)

    def _channel_for(self, source: Any) -> grpc.Channel | None:
        """A channel to ``source``'s learner node (cached per (node, addresses); probed once: the first address that
        answers within ``connect_timeout_sec``); None when none answers (T3.2: then the relay)."""
        from colosseum.node.dataport import probe_direct

        link = self._link
        key = (int(source.node), tuple(source.addresses))
        with self._channels_lock:
            hit = self._channels.get(key)
        if hit is not None:
            return hit[1]
        probed = probe_direct(list(source.addresses), link.make_channel, link.connect_timeout_sec)
        if probed is None:
            return None
        with self._channels_lock:
            stale = [k for k in self._channels if k[0] == key[0] and k != key]
            old = [self._channels.pop(k)[1] for k in stale]
            self._channels[key] = probed
        for channel in old:
            channel.close()
        return probed[1]

    def _drop_channel(self, source: Any) -> None:
        key = (int(source.node), tuple(source.addresses))
        with self._channels_lock:
            hit = self._channels.pop(key, None)
        if hit is not None:
            hit[1].close()

    def _latest_job(self, job: FetchJob, serial: int) -> None:
        """A transfer-pool thread: ``GetLatest`` on the learner node's data port; posts the outcome."""
        from colosseum.node.dataport import FetchError, fetch_latest

        agent, link = job.agent_id, self._link
        if link is None or link.make_channel is None:
            self._post(LatestFetchFailed(agent, "unreachable", link.why_not if link else "no data link", serial))
            return
        if not job.source.addresses:
            self._post(LatestFetchFailed(agent, "unreachable", f"node {job.source.node} announced no data port "
                                         f"address (relay only)", serial))
            return
        try:
            channel = self._channel_for(job.source)
        except Exception as e:  # noqa: BLE001 - reported as a failed fetch, retried after one interval
            self._post(LatestFetchFailed(agent, "unreachable", rpc_text(e), serial))
            return
        if channel is None:
            self._post(LatestFetchFailed(agent, "unreachable", f"no answer from {', '.join(job.source.addresses)} "
                                         f"within {link.connect_timeout_sec:g} s", serial))
            return
        try:
            fetched = fetch_latest(channel, agent, job.since_version, max_message_mb=link.max_message_mb,
                                   metadata=link.metadata, timeout=DOWNLOAD_TIMEOUT_SEC)
        except FetchError as e:
            self._drop_channel(job.source)
            self._post(LatestFetchFailed(agent, e.reason, str(e), serial))
            return
        self._post(LatestFetched(agent, fetched, serial))

    def latest_done(self, run: Any, ev: LatestFetched) -> Literal["published", "kept", "failed"]:
        """Main loop: a latest transfer of the current run ended. A newer version is published into the cache
        (byte for byte the learner's frame, with its ``teacher_active``); ``kept``: not modified or not newer;
        ``failed``: it could not be published (cache full: ERROR with the hint, once per run; a frame the cache
        refuses: WARNING)."""
        from colosseum.node.weights import CacheFullError

        self._latest.pop(ev.agent_id, None)
        planner = self.planner
        cached = run.reader.latest_version(ev.agent_id)
        have = -1 if cached is None else int(cached[0])
        fetched = ev.fetched
        if fetched is None or int(fetched.version) <= have:
            planner.finished(ev.agent_id, None if fetched is None else int(fetched.version), cached=have)
            return "kept"
        try:
            run.writer.publish_latest(ev.agent_id, fetched.frame, version=int(fetched.version),
                                      generation=int(fetched.generation), teacher_active=bool(fetched.teacher_active))
        except CacheFullError as e:
            planner.failed(ev.agent_id)
            if ("cache_full", "") not in self._warned:
                self._warned.add(("cache_full", ""))
                logger.error(f"node: the weights of agent {ev.agent_id} v{fetched.version} do not fit: {e}")
            return "failed"
        except (OSError, ValueError) as e:
            planner.failed(ev.agent_id)
            self._warn(ev.agent_id, "publish", f"cannot put v{fetched.version} into the weight cache ({e})")
            return "failed"
        planner.finished(ev.agent_id, int(fetched.version), cached=have)
        self.fetch_bytes += len(fetched.frame)
        return "published"

    def latest_failed(self, ev: LatestFetchFailed) -> None:
        """Main loop: a latest transfer of the current run failed (retried after one interval; one WARNING per
        (agent, reason) per run)."""
        self._latest.pop(ev.agent_id, None)
        if self.planner is not None:
            self.planner.failed(ev.agent_id)
        source = None if self.planner is None else self.planner.known(ev.agent_id)
        where = f" from node {source.node}" if source is not None else ""
        interval = "" if self.planner is None else f"; retried every {self.planner.interval:g} s"
        self._warn(ev.agent_id, ev.reason, f"latest weights{where}: {ev.reason} ({ev.detail}){interval}")

    def _warn(self, agent: str, reason: str, text: str) -> None:
        if (agent, reason) in self._warned:
            return
        self._warned.add((agent, reason))
        logger.warning(f"node: agent {agent}: {text} (further failures of this kind are only counted)")

    def counters(self) -> dict[str, int]:
        """The fetch counters of this node process (A2 names)."""
        return {**self.stats.as_counters(), "weight_fetch_bytes": int(self.fetch_bytes)}


def rpc_text(e: BaseException) -> str:
    if isinstance(e, grpc.RpcError) and hasattr(e, "code"):
        return f"{e.code().name}: {e.details() if hasattr(e, 'details') else ''}".rstrip(": ")
    return f"{type(e).__name__}: {e}"
