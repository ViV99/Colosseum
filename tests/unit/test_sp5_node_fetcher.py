"""``NodeFetcher``'s data-port channels (SP5 T3.1 review round 1): two transfers that probe the same learner node at
once keep one channel and close the other."""
from __future__ import annotations

import threading
from types import SimpleNamespace

from colosseum.node import dataport
from colosseum.node.fetch import NodeFetcher, _DataLink


class _Channel:
    def __init__(self, name):
        self.name, self.closed = name, False

    def close(self):
        self.closed = True


def test_concurrent_probes_of_one_place_keep_one_channel_and_close_the_other(monkeypatch):
    barrier = threading.Barrier(2)
    made = []

    def probe(addresses, make_channel, timeout):
        barrier.wait(5)                                   # both transfers missed the channel cache
        channel = _Channel(f"c{len(made)}")
        made.append(channel)
        return addresses[0], channel

    monkeypatch.setattr(dataport, "probe_direct", probe)
    fetcher = NodeFetcher(submit=lambda *a: None, post=lambda item: None)
    fetcher._link = _DataLink(lambda a, n: None, (), 1.0, 64)
    source = SimpleNamespace(node=7, addresses=("127.0.0.1:1",))
    got = []
    threads = [threading.Thread(target=lambda: got.append(fetcher._channel_for(source))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(made) == 2 and got[0] is got[1]                # both transfers use the kept channel
    assert sorted(c.closed for c in made) == [False, True]    # the loser is closed, not leaked
    assert not got[0].closed
    fetcher.close_channels()
    assert all(c.closed for c in made)


def _job(mode="announced", node=7, addresses=("127.0.0.1:1",), relay_only=False):
    from colosseum.net.messages import VersionInfo
    from colosseum.node.fetch import FetchJob

    info = VersionInfo(agent_id="main", generation=1, version=4, size=1, digest="", node=node,
                       addresses=tuple(addresses), relay_only=relay_only)
    return FetchJob("main", -1, info, mode)


def _fetcher(monkeypatch, *, relay, probe_answers):
    """A fetcher whose direct probe answers or not, whose hub ``Download(kind="latest")`` returns a marker."""
    posted, calls = [], []
    monkeypatch.setattr(dataport, "probe_direct",
                        lambda addresses, make_channel, timeout: (addresses[0], _Channel("d")) if probe_answers
                        else None)
    monkeypatch.setattr(dataport, "fetch_latest", lambda *a, **k: (calls.append("direct"), None)[1])
    monkeypatch.setattr(dataport, "fetch_latest_relay", lambda *a, **k: (calls.append("relay"), None)[1])
    fetcher = NodeFetcher(submit=lambda *a: None, post=posted.append)
    fetcher.start_run(number=2, sync_interval_sec=1.0, link=_DataLink(lambda a, n: _Channel(a), (), 1.0, 64),
                      relay=relay)
    fetcher.set_hub(_Channel("hub"), ())
    return fetcher, posted, calls


def test_latest_weights_come_through_the_hub_only_when_the_data_port_fails_and_the_relay_is_on(monkeypatch):
    """T3.2 (spec 4.8): direct first; a probe without an answer -> ``Download(kind="latest")`` when the relay is on
    (and the direct path is not probed again for ``DIRECT_RETRY_SEC``); relay off -> ``unreachable``; a relay-only
    source -> through the hub at once; parked weights (node -1) -> the hub whatever the relay switch (P-PF3-13); poll
    mode (no hub) -> never the relay."""
    from colosseum.node.fetch import LatestFetched, LatestFetchFailed

    fetcher, posted, calls = _fetcher(monkeypatch, relay=True, probe_answers=True)
    fetcher._latest_job(_job(), 1)
    assert calls == ["direct"] and isinstance(posted[-1], LatestFetched)

    fetcher, posted, calls = _fetcher(monkeypatch, relay=True, probe_answers=False)
    fetcher._latest_job(_job(), 1)
    fetcher._latest_job(_job(), 1)                                   # node 7 failed recently: no second probe
    assert calls == ["relay", "relay"] and all(isinstance(p, LatestFetched) for p in posted)
    fetcher._latest_job(_job(relay_only=True, addresses=()), 1)
    assert calls[-1] == "relay"

    fetcher, posted, calls = _fetcher(monkeypatch, relay=False, probe_answers=False)
    fetcher._latest_job(_job(), 1)
    assert calls == [] and isinstance(posted[-1], LatestFetchFailed) and posted[-1].reason == "unreachable"
    fetcher._latest_job(_job(node=-1, addresses=(), relay_only=True), 1)
    assert calls == ["relay"]                                         # parked: hub-held weights
    fetcher._latest_job(_job(mode="poll"), 1)
    assert calls == ["relay"] and posted[-1].reason == "unreachable"


def test_relayed_weights_not_on_the_hub_yet_are_a_quiet_retry(monkeypatch, caplog):
    import logging

    from colosseum.node.fetch import LatestFetchFailed

    fetcher, posted, _calls = _fetcher(monkeypatch, relay=True, probe_answers=False)

    def pending(*a, **k):
        raise dataport.FetchError("relay_pending", "UNAVAILABLE: not on the hub yet")
    monkeypatch.setattr(dataport, "fetch_latest_relay", pending)
    fetcher._latest_job(_job(), 1)
    assert isinstance(posted[-1], LatestFetchFailed) and posted[-1].reason == "relay_pending"
    with caplog.at_level(logging.WARNING, logger="colosseum.node.fetch"):
        fetcher.latest_failed(posted[-1])
    assert not caplog.records and fetcher.stats.failed == 1           # counted, no WARNING
