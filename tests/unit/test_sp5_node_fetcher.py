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
