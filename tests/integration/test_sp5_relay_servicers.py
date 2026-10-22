"""The hub relay's servicers (SP5 T3.2, spec 4.8): ``RelaySend`` -> pair buffers -> ``RelayInbox`` with credits,
frames byte-identical and counted; frames for an offline node dropped and counted; relayed latest weights pulled once
and uploaded once (``Upload(kind="relay_latest")``, no ``UploadIn``) for many ``Download(kind="latest")`` calls;
uploads from another node or with a wrong digest refused; ``distributed.relay: false`` refuses the relay streams.

The real hub server of ``build_server`` with ``hub_kit`` fake nodes as in-process clients; the test publishes the
online nodes and the versions itself, standing in for the hub main loop."""
from __future__ import annotations

import queue
import threading

import grpc
import numpy as np
import pytest

from colosseum.hub.relay import LatestRelay, RelayHub
from colosseum.hub.server import UploadIn
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.grpc_util import blob_digest, send_parts
from colosseum.net.messages import VersionInfo
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.node.dataport import read_latest_reply
from hub_kit import serve

MAX_MB = 64


@pytest.fixture
def relay_hub(tmp_path):
    def make(enabled=True):
        relay = RelayHub(enabled=enabled)
        latest = LatestRelay(tmp_path / "run" / "blobs" / "relay", relay, wait_sec=10.0,
                             make_pull=lambda agent, since, rid: pb.LatestRequest(agent_id=agent, since_version=since,
                                                                                  request_id=rid))
        hub = serve(tmp_path, relay=relay, latest=latest)
        nodes = [hub.registered_node(machine_id=f"m{i}", name=f"n{i}") for i in range(4)]
        relay.set_online({n.node_id: n.number for n in nodes})
        made.append(hub)
        return hub, relay, latest, nodes
    made: list = []
    yield make
    for hub in made:
        hub.stop()


class Inbox:
    """A node's RelayInbox stream: ``initial`` credits, then one per item read (``grant=False``: none)."""

    def __init__(self, node, initial=16, grant=True):
        self.items: list = []
        self._credits: queue.Queue = queue.Queue()
        self._credits.put(pb.RelayCredit(frames=initial, node_id=node.node_id))
        self._grant = grant

        def credits():
            while True:
                item = self._credits.get()
                if item is None:
                    return
                yield item

        self.call = node.stub().RelayInbox(credits(), metadata=node._node_md())
        self.error: list = []
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for item in self.call:
                self.items.append(item)
                if self._grant and item.WhichOneof("i") == "chunk":
                    self._credits.put(pb.RelayCredit(frames=1))
                self.on_item(item)
        except grpc.RpcError as e:
            self.error.append(e.code())

    def on_item(self, item):
        pass

    def close(self):
        self._credits.put(None)
        self.call.cancel()


def _wait(pred, timeout=10.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def _frames(target, n, size=1000):
    return [pb.ChunkFrame(agent_id="main", target_node=target, generation=1, frame=bytes([k % 251]) * (size + k))
            for k in range(n)]


def test_relay_streams_frames_to_the_target_with_credits(relay_hub):
    hub, relay, _latest, (n1, n2, *_rest) = relay_hub()
    inbox = Inbox(n2)
    frames = _frames(n2.number, 50)
    summary = n1.stub().RelaySend(iter(frames), metadata=n1._node_md(), timeout=30)
    assert (summary.received, summary.dropped) == (50, 0)
    assert _wait(lambda: len(inbox.items) == 50)
    assert [i.chunk for i in inbox.items] == frames                     # byte-identical, in order per source
    stats = relay.stats()
    assert stats["relay_frames"] == 50 and stats["relay_bytes"] == sum(len(f.frame) for f in frames)
    inbox.close()


def test_frames_for_an_offline_target_are_dropped_and_counted(relay_hub):
    hub, relay, _latest, (n1, *_rest) = relay_hub()
    summary = n1.stub().RelaySend(iter(_frames(9, 7)), metadata=n1._node_md(), timeout=30)
    assert (summary.received, summary.dropped) == (7, 7)
    assert relay.stats()["dropped_chunks.disconnect"] == 7


def _weights(version):
    return encode_weights({"w": np.full((300_000,), float(version), np.float32)},
                          WeightsMeta(agent_id="main", kind="latest", generation=1, policy_version=version))


def _upload(node, frame, *, version=5, digest=None):
    header = pb.UploadHeader(kind="relay_latest", agent_id="main", generation=1, version=version,
                             total_size=len(frame), digest=digest or blob_digest(frame), request_id="r")

    def parts():
        yield pb.UploadPart(header=header)
        for part in send_parts(frame):
            if part.data:
                yield pb.UploadPart(data=part.data)
    return node.stub().Upload(parts(), metadata=node._node_md(), timeout=30)


def test_relayed_latest_weights_are_uploaded_once_for_many_downloads(relay_hub, tmp_path):
    hub, relay, latest, (n1, n2, n3, n4) = relay_hub()
    latest.publish_versions({"main": VersionInfo(agent_id="main", generation=1, version=5, size=1, digest="d",
                                                 node=n2.number, relay_only=True)})
    frame = _weights(5)
    pulls, replies = [], []
    inbox = Inbox(n2)
    inbox.on_item = lambda item: (pulls.append(item.pull), replies.append(_upload(n2, frame))) \
        if item.WhichOneof("i") == "pull" else None
    got = []

    def download(node):
        parts = node.stub().Download(pb.BlobRequest(kind="latest", agent_id="main", since_version=-1),
                                     metadata=node._node_md(), timeout=30)
        got.append(read_latest_reply(parts, "main", max_message_mb=MAX_MB, source="relay"))
    threads = [threading.Thread(target=download, args=(n,)) for n in (n1, n3, n4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(got) == 3 and all(g.frame == frame and g.version == 5 and g.source == "relay" for g in got)
    assert len(pulls) == 1 and replies[0].accepted                       # one pull, one upload
    assert len(list((tmp_path / "run" / "blobs" / "relay").glob("latest-*-v5.frame"))) == 1
    assert hub.loop.events(UploadIn) == []                               # no UploadIn: no hub state touched
    not_modified = n1.stub().Download(pb.BlobRequest(kind="latest", agent_id="main", since_version=5),
                                      metadata=n1._node_md(), timeout=30)
    assert read_latest_reply(not_modified, "main", max_message_mb=MAX_MB, source="relay") is None
    refused = _upload(n1, _weights(6), version=6)                        # not the learner's node
    assert (refused.accepted, refused.reason) == (False, "not_learner_node")
    bad = _upload(n2, _weights(6), version=6, digest="0" * 32)
    assert (bad.accepted, bad.reason) == (False, "digest")
    assert [p.name for p in (tmp_path / "run" / "blobs" / "relay").iterdir()] == [
        p.name for p in (tmp_path / "run" / "blobs" / "relay").glob("latest-*-v5.frame")]
    assert latest.stats()["relay_uploads"] == 1 and latest.stats()["relay_uploads_refused"] == 2
    inbox.close()


def test_relay_off_refuses_relay_streams(relay_hub):
    hub, relay, _latest, (n1, n2, *_rest) = relay_hub(enabled=False)
    with pytest.raises(grpc.RpcError) as e:
        n1.stub().RelaySend(iter(_frames(n2.number, 1)), metadata=n1._node_md(), timeout=10)
    assert e.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    inbox = Inbox(n2)
    assert _wait(lambda: inbox.error == [grpc.StatusCode.FAILED_PRECONDITION])
