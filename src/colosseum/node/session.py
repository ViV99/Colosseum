"""The node's session client (SP5 spec 4.1, 3.2-3.3): one ``Hub/Session`` stream to the hub at a time, reconnected
with backoff until ``close``.

``hello()`` is called at every (re)connect (it carries the node's live cells and last acked seq); every ``HubMessage``
goes to ``inbox`` as it is, with ``SessionUp`` before the first one of each stream (the stream is up: the hub took
the ``Hello``; the next message is the ``Welcome``) and ``SessionLost`` after each stream or failed attempt (its
``reason`` names the gRPC status). ``send`` puts a message on the current stream (False while no stream is up: the
caller keeps it, e.g. the result buffer resends after the next ``SessionUp``).

Channels: ``grpc.secure_channel`` with the pinned hub CA (``hub_channel_credentials``) and the fixed server name
``colosseum-hub``, or a plain channel with ``credentials=None`` (``--insecure`` and the local loopback port); the node
id and node secret go in the call metadata in every mode (R1). A new channel per attempt (a broken one is never
reused).
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import grpc

from colosseum.net.grpc_util import channel_options, node_metadata
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc
from colosseum.net.security import HUB_SERVER_NAME

__all__ = ["HUB_LOST_RETRY_SEC", "RECONNECT_MAX_SEC", "RECONNECT_MIN_SEC", "HubSession", "SessionLost", "SessionUp",
           "backoff_delays"]

log = logging.getLogger(__name__)

RECONNECT_MIN_SEC, RECONNECT_MAX_SEC = 0.5, 10.0          # spec 4.1 backoff (doubling, jitter <= 20 %)
HUB_LOST_RETRY_SEC = (30.0, 60.0)    # spec 7.1: retry pause after hub_lost_timeout_sec (T4.4)
OUTBOUND_QUEUE_SIZE = 10_000         # node -> hub messages waiting on the current stream
CLOSE_TIMEOUT_SEC = 5.0              # close(): the stream ends by itself, else it is cancelled
POLL_SEC = 0.25
_JITTER = 0.2
_END = object()


@dataclass
class SessionUp:
    """The stream is open and the Hello was taken (the next HubMessage is the Welcome)."""


@dataclass
class SessionLost:
    reason: str


def backoff_delays(rng: random.Random) -> Iterator[float]:
    """0.5, 1, 2, ... 10 s (spec 4.1), each with up to 20 % jitter either way."""
    delay = RECONNECT_MIN_SEC
    while True:
        yield delay * (1.0 + rng.uniform(-_JITTER, _JITTER))
        delay = min(RECONNECT_MAX_SEC, delay * 2.0)


class _Stream:
    """One session attempt: its outbound queue, whether its first response arrived, its call."""

    def __init__(self) -> None:
        self.out: queue.Queue = queue.Queue(maxsize=OUTBOUND_QUEUE_SIZE)
        self.ended = threading.Event()
        self.connected = False
        self.call = None

    def requests(self) -> Iterator[pb.NodeMessage]:
        while True:
            try:
                item = self.out.get(timeout=POLL_SEC)
            except queue.Empty:
                if self.ended.is_set():
                    return
                continue
            if item is _END:
                return
            yield item

    def end(self) -> None:
        """Half-close the request stream (the hub then ends the session)."""
        self.connected = False
        self.ended.set()
        try:
            self.out.put_nowait(_END)
        except queue.Full:                       # the generator sees ``ended`` once the queue drains
            pass


class HubSession:
    """Session client (contract; module docstring). One thread runs connect -> stream -> on loss wait the next backoff
    delay -> reconnect, until ``close``; the backoff restarts after a stream that stayed up for
    ``RECONNECT_MAX_SEC``."""

    def __init__(self, address: str, *, credentials: grpc.ChannelCredentials | None, node_id: str, node_secret: str,
                 hello: Callable[[], pb.Hello], inbox: Callable[[object], None], max_message_mb: int,
                 rng: random.Random | None = None) -> None:
        self._address = address
        self._credentials = credentials
        self._metadata = node_metadata(node_id, node_secret)
        self._hello = hello
        self._inbox = inbox
        self._max_message_mb = int(max_message_mb)
        self._rng = rng or random.Random()
        self._closing = threading.Event()
        self._stream: _Stream | None = None
        self._thread = threading.Thread(target=self._run, name="hub-session", daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def connected(self) -> bool:
        """A stream is up and its first response arrived."""
        stream = self._stream
        return stream is not None and stream.connected

    def send(self, msg: pb.NodeMessage) -> bool:
        """Queue ``msg`` on the current stream; False while no stream is up (or its queue is full)."""
        stream = self._stream
        if stream is None or not stream.connected or self._closing.is_set():
            return False
        try:
            stream.out.put_nowait(msg)
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        """End the stream (outbound sentinel), stop the thread (cancelling a stream that does not end within
        ``CLOSE_TIMEOUT_SEC``), close the channel. No reconnect afterwards."""
        self._closing.set()
        stream = self._stream
        if stream is not None:
            stream.end()
        if not self._thread.is_alive():
            return
        self._thread.join(CLOSE_TIMEOUT_SEC / 2)
        stream = self._stream
        if self._thread.is_alive() and stream is not None and stream.call is not None:
            stream.call.cancel()
        self._thread.join(CLOSE_TIMEOUT_SEC / 2)

    # ---- the session thread -------------------------------------------------------------------------------------

    def _deliver(self, item: object) -> None:
        try:
            self._inbox(item)
        except Exception:  # noqa: BLE001 - the node's inbox must not stop the session thread
            log.exception("the node inbox refused %s", type(item).__name__)

    def _channel(self) -> grpc.Channel:
        if self._credentials is None:
            return grpc.insecure_channel(self._address, options=channel_options(self._max_message_mb))
        return grpc.secure_channel(self._address, self._credentials,
                                   options=channel_options(self._max_message_mb, server_name=HUB_SERVER_NAME))

    def _run(self) -> None:
        delays = backoff_delays(self._rng)
        while not self._closing.is_set():
            up_for = self._run_stream()
            if self._closing.is_set():
                break
            if up_for >= RECONNECT_MAX_SEC:
                delays = backoff_delays(self._rng)
            self._closing.wait(next(delays))

    def _run_stream(self) -> float:
        """One attempt: returns how long the stream was up (0 if it never was)."""
        try:
            hello = self._hello()
        except Exception as e:  # noqa: BLE001 - reported as a lost attempt; the next one calls hello() again
            log.exception("building the Hello failed")
            self._deliver(SessionLost(f"hello failed: {type(e).__name__}: {e}"))
            return 0.0
        stream = _Stream()
        stream.out.put(pb.NodeMessage(hello=hello))
        channel = self._channel()
        self._stream = stream
        up_since: float | None = None
        reason = "end of stream"
        try:
            call = pb_grpc.HubStub(channel).Session(stream.requests(), metadata=self._metadata)
            stream.call = call
            if self._closing.is_set():
                call.cancel()
            for msg in call:
                if up_since is None:
                    up_since = time.monotonic()
                    stream.connected = not self._closing.is_set()
                    self._deliver(SessionUp())
                self._deliver(msg)
        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            details = e.details() if hasattr(e, "details") else None
            reason = "stream cut" if code is None else f"{code.name}: {details or ''}".rstrip(": ")
        except Exception as e:  # noqa: BLE001 - one broken stream never stops the session thread
            reason = f"{type(e).__name__}: {e}"
        finally:
            stream.end()
            self._stream = None
            channel.close()
        if self._closing.is_set():
            reason = f"closed ({reason})"
        self._deliver(SessionLost(reason))
        return 0.0 if up_since is None else time.monotonic() - up_since
