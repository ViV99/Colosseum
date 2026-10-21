"""The node's session client ``HubSession`` (SP5 T2.3; spec 4.1, 3.2-3.3) against the real hub server and
``hub_kit.MiniHubLoop``: connect and deliver hub messages, reconnect with backoff and the same session id, close,
a pinned CA, the node secret on an insecure hub."""
from __future__ import annotations

import logging
import random
import threading
import time

import pytest

from colosseum.hub.server import NodeMessageIn, SessionClosed, SessionOpened
from colosseum.net.messages import PROTOCOL_VERSION
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.security import HubIdentity, hub_channel_credentials
from colosseum.node import session as node_session
from colosseum.node.session import HubSession, SessionLost, SessionUp, backoff_delays
from hub_kit import poll_until, serve


@pytest.fixture
def hub(tmp_path):
    served = serve(tmp_path)
    try:
        yield served
    finally:
        served.stop()


class Client:
    """A HubSession of a registered node with a counting ``hello`` and a list inbox. ``gate``: when set, every
    ``hello`` after the first waits for it (holds a reconnect while the test looks at the disconnected client)."""

    def __init__(self, hub, *, credentials="hub", node=None, secret=None, session_id="s1", gate=None,
                 inbox=None):
        self.node = node or hub.registered_node()
        self.session_id = session_id
        self.hellos = 0
        self.gate: threading.Event | None = gate
        self.inbox: list = []
        creds = hub_channel_credentials(hub.identity.ca_pem) if credentials == "hub" else credentials
        self.session = HubSession(hub.address, credentials=creds, node_id=self.node.node_id,
                                  node_secret=secret or self.node.node_secret, hello=self.hello,
                                  inbox=inbox or self.inbox.append, max_message_mb=64)

    def hello(self) -> pb.Hello:
        self.hellos += 1
        if self.gate is not None and self.hellos > 1:
            assert self.gate.wait(10), "the test never opened the reconnect gate"
        return pb.Hello(node_id=self.node.node_id, session_id=self.session_id, protocol_major=PROTOCOL_VERSION[0],
                        protocol_minor=PROTOCOL_VERSION[1])

    def of(self, cls) -> list:
        return [m for m in list(self.inbox) if isinstance(m, cls)]

    def hub_messages(self, kind: str) -> list:
        return [m for m in list(self.inbox) if isinstance(m, pb.HubMessage) and m.WhichOneof("m") == kind]


def test_the_session_connects_and_delivers_hub_messages_to_the_node_inbox(hub):
    c = Client(hub)
    try:
        c.session.start()
        poll_until(lambda: c.hub_messages("welcome"))
        assert isinstance(c.inbox[0], SessionUp)
        assert isinstance(c.inbox[1], pb.HubMessage) and c.inbox[1].welcome.node_number == c.node.number
        assert c.session.connected
        assert c.session.send(pb.NodeMessage(heartbeat=pb.Heartbeat(env_steps=10, counters={"worker_restarts": 1})))
        beat = poll_until(lambda: hub.loop.messages("heartbeat"))[0]
        assert beat.node_id == c.node.node_id and beat.message.heartbeat.env_steps == 10
        assert beat.message.heartbeat.counters == {"worker_restarts": 1}
    finally:
        c.session.close()


def test_the_session_reconnects_with_backoff_and_the_same_session_id(hub, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    gate = threading.Event()
    c = Client(hub, gate=gate)
    try:
        c.session.start()
        poll_until(lambda: c.hub_messages("welcome"))
        hub.loop.close_session(c.node.node_id, "test")
        poll_until(lambda: c.of(SessionLost))
        assert hub.loop.wait_events(SessionClosed)[0].reason == "test"
        assert not c.session.connected                           # the reconnect waits in hello() for the gate
        assert not c.session.send(pb.NodeMessage(heartbeat=pb.Heartbeat(env_steps=1)))
        gate.set()
        poll_until(lambda: len(c.of(SessionUp)) >= 2)
        poll_until(lambda: len(c.hub_messages("welcome")) >= 2)
        assert c.hellos == 2
        opened = hub.loop.wait_events(SessionOpened, 2)
        assert [e.hello.session_id for e in opened] == ["s1", "s1"] and hub.loop.connects == ["new", "resumed"]
        assert c.session.send(pb.NodeMessage(heartbeat=pb.Heartbeat(env_steps=2)))
        poll_until(lambda: hub.loop.messages("heartbeat"))
        assert [e.message.heartbeat.env_steps for e in hub.loop.messages("heartbeat")] == [2]
    finally:
        gate.set()
        c.session.close()


def test_backoff_doubles_with_bounded_jitter():
    gen = backoff_delays(random.Random(0))
    delays = [next(gen) for _ in range(8)]
    for k, delay in enumerate(delays):
        base = min(0.5 * 2**k, 10.0)
        assert base * 0.8 <= delay <= base * 1.2
        assert delay <= 10.0 * 1.2
    assert len(set(delays)) == 8                                  # jittered


def test_close_stops_reconnecting(hub, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    c = Client(hub)
    c.session.start()
    poll_until(lambda: c.hub_messages("welcome"))
    c.session.close()
    assert not c.session.connected and not c.session._thread.is_alive()
    time.sleep(1.0)
    assert len(hub.loop.events(SessionOpened)) == 1 and c.hellos == 1
    assert not c.session.send(pb.NodeMessage(heartbeat=pb.Heartbeat()))


def test_close_from_the_inbox_callback_stops_the_session(hub, caplog):
    """``close()`` called on the session thread (from the node's inbox callback) does not try to join itself: it
    returns, the thread stops and the last ``SessionLost`` says closed."""
    holder: dict = {}
    inbox: list = []
    raised: list = []

    def on_message(item):
        inbox.append(item)
        if isinstance(item, pb.HubMessage) and item.WhichOneof("m") == "welcome":
            try:
                holder["client"].session.close()
            except Exception as e:  # noqa: BLE001 - recorded for the assertion
                raised.append(e)

    with caplog.at_level(logging.ERROR, logger="colosseum.node.session"):
        holder["client"] = c = Client(hub, inbox=on_message)
        try:
            c.session.start()
            poll_until(lambda: [m for m in inbox if isinstance(m, SessionLost)])
            poll_until(lambda: not c.session._thread.is_alive(), timeout=5.0)
        finally:
            c.session.close()
    assert raised == [] and caplog.records == []
    lost = [m for m in inbox if isinstance(m, SessionLost)]
    assert len(lost) == 1 and lost[0].reason.startswith("closed")
    assert not c.session.connected and len(hub.loop.events(SessionOpened)) == 1


def test_a_tls_session_refuses_a_hub_with_another_ca(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.1)
    other = HubIdentity.load_or_create(tmp_path / "other-hub")
    c = Client(hub, credentials=hub_channel_credentials(other.ca_pem))
    try:
        c.session.start()
        poll_until(lambda: len(c.of(SessionLost)) >= 2)
        assert all("UNAVAILABLE" in m.reason for m in c.of(SessionLost))
        assert c.of(SessionUp) == [] and not c.session.connected
        assert [e for e in hub.loop.received if isinstance(e, (SessionOpened, NodeMessageIn))] == []
    finally:
        c.session.close()


def test_an_insecure_hub_still_requires_the_node_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    hub = serve(tmp_path, insecure=True)
    wrong = right = None
    try:
        wrong = Client(hub, credentials=None, secret="x" * 43)
        wrong.session.start()
        poll_until(lambda: wrong.of(SessionLost))
        assert "UNAUTHENTICATED" in wrong.of(SessionLost)[0].reason and wrong.of(SessionUp) == []
        wrong.session.close()
        right = Client(hub, credentials=None, node=wrong.node)
        right.session.start()
        poll_until(lambda: right.hub_messages("welcome"))
        assert right.session.connected
    finally:
        for c in (wrong, right):
            if c is not None:
                c.session.close()
        hub.stop()
