"""RelayHub and LatestRelay (SP5 T3.2, spec 4.8): pair buffers with end-to-end credit flow, drops only for an offline
target (disconnect) or a receiver that stopped draining (overflow), round-robin over sources, weight pulls first
and without credit, frames lost inside a broken inbox counted at detach (pre-flight ruling P-PF3-14); relayed latest
weights uploaded once per version."""
from __future__ import annotations

import threading
import time

from colosseum.hub.relay import LatestRelay, RelayHub
from colosseum.hub.state import VersionInfo


def _hub(**kw):
    hub = RelayHub(**kw)
    hub.set_online({"src": 1, "src2": 3, "dst": 2})
    return hub


def test_a_frame_waits_for_a_credit_and_is_counted_when_sent():
    hub = _hub()
    assert hub.put(1, 2, "f1", 100) is None
    inbox = hub.attach(2)
    assert hub.next_item(inbox, timeout=0.05) is None             # no credit yet
    hub.add_credits(inbox, 1)
    assert hub.next_item(inbox, timeout=0.05) == ("chunk", "f1")
    assert hub.stats() == {"relay_frames": 1, "relay_bytes": 100}


def test_a_full_pair_buffer_blocks_the_sender_until_the_receiver_grants_credit():
    hub = _hub(pair_frames=2, put_timeout_sec=5.0)
    inbox = hub.attach(2)
    assert hub.put(1, 2, "a", 1) is None and hub.put(1, 2, "b", 1) is None
    done = threading.Event()
    threading.Thread(target=lambda: (hub.put(1, 2, "c", 1), done.set()), daemon=True).start()
    assert not done.wait(0.3)                                       # backpressure: the third put waits
    hub.add_credits(inbox, 1)
    assert hub.next_item(inbox, timeout=1.0) == ("chunk", "a")
    assert done.wait(2.0)
    assert hub.stats().get("dropped_chunks.overflow", 0) == 0


def test_a_single_large_frame_is_admitted_into_an_empty_buffer():
    hub = _hub(pair_bytes=10)
    assert hub.put(1, 2, "big", 1000) is None
    assert hub.put(1, 2, "more", 1, should_stop=lambda: True) == "disconnect"   # full: the stream is ending


def test_overflow_when_the_receiver_stops_draining():
    t = [0.0]
    hub = _hub(pair_frames=1, put_timeout_sec=1.0, clock=lambda: t[0])
    assert hub.put(1, 2, "a", 1) is None

    def advance():
        time.sleep(0.05)
        t[0] = 2.0
    threading.Thread(target=advance, daemon=True).start()
    assert hub.put(1, 2, "b", 1) == "overflow"
    assert hub.stats()["dropped_chunks.overflow"] == 1


def test_frames_for_an_offline_node_are_dropped_and_counted_disconnect():
    hub = _hub()
    assert hub.put(1, 9, "x", 1) == "disconnect"
    assert hub.put(1, 2, "a", 1) is None and hub.put(1, 2, "b", 1) is None
    inbox = hub.attach(2)
    hub.set_online({"src": 1})                                     # node 2 lost: its buffers go
    assert inbox.detached
    assert hub.stats()["dropped_chunks.disconnect"] == 3
    assert hub.next_item(inbox, timeout=0.05) is None


def test_frames_in_flight_to_a_broken_inbox_are_counted_at_detach():
    """P-PF3-14: frames sent down an inbox but never credited back (the receiver died, the stream broke) are counted
    ``disconnect`` when the inbox detaches (an upper bound)."""
    hub = _hub()
    for i in range(5):
        hub.put(1, 2, f"f{i}", 10)
    inbox = hub.attach(2, initial_credits=4)
    assert [hub.next_item(inbox, timeout=0.1)[1] for _ in range(4)] == ["f0", "f1", "f2", "f3"]
    hub.add_credits(inbox, 1)                                       # one frame delivered and credited back
    assert hub.next_item(inbox, timeout=0.1) == ("chunk", "f4")
    hub.detach(inbox)
    hub.detach(inbox)                                               # counted once
    assert hub.stats()["dropped_chunks.disconnect"] == 4            # 5 sent, 1 credited back


def test_sources_are_served_round_robin():
    hub = _hub()
    for i in range(3):
        hub.put(1, 2, f"s1-{i}", 1)
    hub.put(3, 2, "s3-0", 1)
    inbox = hub.attach(2)
    hub.add_credits(inbox, 4)
    got = [hub.next_item(inbox, timeout=0.1)[1] for _ in range(4)]
    assert got == ["s1-0", "s3-0", "s1-1", "s1-2"]


def test_controls_go_first_and_need_no_credit_and_a_new_inbox_detaches_the_old():
    hub = _hub()
    assert hub.push_control(2, "pull") is False                     # no inbox attached
    old = hub.attach(2)
    hub.put(1, 2, "a", 1)
    assert hub.push_control(2, "pull") is True
    assert hub.next_item(old, timeout=0.05) == ("control", "pull")
    new = hub.attach(2)
    assert old.detached and not new.detached
    hub.detach(new)
    assert hub.push_control(2, "pull") is False


def _vi(version, generation=1, node=2):
    return VersionInfo(agent_id="main", generation=generation, version=version, size=4, digest="d", node=node,
                       relay_only=True)


def _files(root):
    return sorted(p.name for p in root.iterdir())


def test_latest_relay_pulls_once_and_serves_every_waiter_from_one_upload(tmp_path):
    hub = _hub()
    inbox = hub.attach(2)
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=5.0)
    relay.publish_versions({"main": _vi(7)})
    results = []

    def download():
        got = relay.open_newer("main", -1)
        if got is not None:
            meta, f = got
            with f:
                results.append((meta.version, f.read()))
    threads = [threading.Thread(target=download) for _ in range(3)]
    for th in threads:
        th.start()
    kind, pull = hub.next_item(inbox, timeout=2.0)
    assert kind == "control" and pull[0] == "main" and pull[1] == -1
    assert hub.next_item(inbox, timeout=0.2) is None                # one pull for three downloads
    spool = tmp_path / "spool"
    spool.write_bytes(b"w7w7")
    assert relay.complete("main", 7, 1, spool, digest="dg") == "accepted"
    for th in threads:
        th.join(5)
    assert results == [(7, b"w7w7")] * 3
    assert relay.open_newer("main", 7, timeout=0.1) is None         # nothing newer is known
    [name] = _files(tmp_path / "relay")
    assert name.startswith("latest-") and name.endswith("-v7.frame") and "main" not in name   # no agent id in names
    assert relay.stats() == {"relay_uploads": 1}


def test_latest_relay_keeps_only_the_newest_and_refuses_stale_uploads(tmp_path):
    hub = _hub()
    hub.attach(2)
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=0.1)
    relay.publish_versions({"main": _vi(8, generation=2)})
    for name, version, generation, expected in (("a", 7, 1, "stale_generation"), ("b", 8, 2, "accepted"),
                                                 ("c", 8, 2, "not_newer")):
        p = tmp_path / name
        p.write_bytes(b"x")
        assert relay.complete("main", version, generation, p) == expected
        assert not p.exists()
    relay.publish_versions({"main": _vi(9, generation=2)})
    p = tmp_path / "d"
    p.write_bytes(b"y")
    assert relay.complete("main", 9, 2, p) == "accepted"
    [name] = _files(tmp_path / "relay")
    assert name.endswith("-v9.frame")
    assert relay.stats() == {"relay_uploads": 2, "relay_uploads_refused": 2}


def test_latest_relay_gives_up_without_a_learner_inbox(tmp_path):
    hub = _hub()
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=5.0)
    relay.publish_versions({"main": _vi(3)})
    assert relay.open_newer("main", -1) is None                      # node 2 has no RelayInbox: no wait
    assert relay.open_newer("other", -1) is None


def test_a_vanished_relayed_file_is_pulled_again_never_spun_on(tmp_path):
    """Review round 1 of T3.2(b): a served file removed outside the hub (a tmp cleaner) is forgotten and pulled again
    within the timeout; ``stats()`` / ``publish_versions`` are never blocked by a spinning ``open_newer``."""
    hub = _hub()
    inbox = hub.attach(2)
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=5.0)
    relay.publish_versions({"main": _vi(7)})
    spool = tmp_path / "spool"
    spool.write_bytes(b"w7")
    assert relay.complete("main", 7, 1, spool) == "accepted"
    [path] = (tmp_path / "relay").iterdir()
    path.unlink()                                                   # removed outside the hub
    out = []
    started = time.monotonic()
    worker = threading.Thread(target=lambda: out.append(relay.open_newer("main", -1, timeout=1.0)), daemon=True)
    worker.start()
    kind, pull = hub.next_item(inbox, timeout=2.0)                  # pulled again
    assert kind == "control" and pull[0] == "main"
    assert relay.stats()["relay_files_lost"] == 1                   # not blocked meanwhile
    relay.publish_versions({"main": _vi(7)})
    worker.join(5)
    assert out == [None] and time.monotonic() - started < 4.0       # within its timeout, no spin


def test_open_newer_ends_when_its_caller_stops_waiting(tmp_path):
    """Review round 1 of T3.2(b): the wait for a pull ends with the RPC (``should_stop``), not after ``wait_sec``."""
    hub = _hub()
    hub.attach(2)
    relay = LatestRelay(tmp_path / "relay", hub, wait_sec=30.0)
    relay.publish_versions({"main": _vi(7)})
    stop = threading.Event()
    out = []
    worker = threading.Thread(target=lambda: out.append(relay.open_newer("main", -1, should_stop=stop.is_set)),
                              daemon=True)
    worker.start()
    time.sleep(0.2)
    stop.set()
    started = time.monotonic()
    worker.join(5)
    assert out == [None] and time.monotonic() - started < 2.0
