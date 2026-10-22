"""FetchPlanner (SP5 T3.1, spec 4.5, 4.6, criterion 4): which latest weights a node fetches from other machines and
when. One transfer per agent in flight, one download per new version, the sync cadence, first weights at once,
GetLatest polls only without the hub and only to direct sources, never the node's own learners."""
from __future__ import annotations

import pytest

from colosseum.hub.state import VersionInfo
from colosseum.node.fetch import FetchPlanner


def _info(agent="main", version=1, *, node=2, generation=1, addresses=("10.0.0.2:47102",), relay_only=False):
    return VersionInfo(agent_id=agent, generation=generation, version=version, size=10, digest="d", node=node,
                       addresses=addresses, relay_only=relay_only)


def _planner(**kw):
    return FetchPlanner(node=1, sync_interval_sec=kw.pop("interval", 5.0))


def test_first_weights_are_fetched_at_once_and_once():
    p = _planner()
    p.on_version(_info(version=3))
    jobs = p.due({}, hub_online=True, now=0.0)
    assert [(j.agent_id, j.since_version, j.mode, j.source.version) for j in jobs] == [("main", -1, "announced", 3)]
    assert p.in_flight("main")
    assert p.due({}, hub_online=True, now=0.1) == []             # one transfer per agent in flight
    assert p.finished("main", 3, cached=-1) is True
    assert p.stats.history == {"main": [3]} and p.stats.fetched == 1


def test_a_newer_version_is_fetched_once_per_sync_interval_while_referenced():
    p = _planner(interval=5.0)
    p.set_referenced({"main"})
    p.on_version(_info(version=1))
    [job] = p.due({}, hub_online=True, now=0.0)
    p.finished("main", 1, cached=-1)
    for v in (2, 3, 4):
        p.on_version(_info(version=v))
    assert p.due({"main": 1}, hub_online=True, now=4.9) == []    # cadence: one check per interval
    [job] = p.due({"main": 1}, hub_online=True, now=5.0)
    assert job.source.version == 4 and job.since_version == 1    # only the newest is fetched
    p.finished("main", 4, cached=1)
    assert p.due({"main": 4}, hub_online=True, now=20.0) == []   # nothing newer announced
    assert p.stats.history == {"main": [1, 4]}


def test_an_unreferenced_agent_is_fetched_only_until_something_is_cached():
    p = _planner()
    p.on_version(_info(version=1))
    [_] = p.due({}, hub_online=True, now=0.0)
    p.finished("main", 1, cached=-1)
    p.on_version(_info(version=9))
    assert p.due({"main": 1}, hub_online=True, now=100.0) == []
    p.set_referenced({"main"})
    assert [j.source.version for j in p.due({"main": 1}, hub_online=True, now=100.0)] == [9]


def test_own_learners_are_never_fetched():
    p = _planner()
    p.set_referenced({"main", "other"})
    p.on_version(_info("main", 5, node=1))                       # the learner of main runs on this node (1)
    p.on_version(_info("other", 5, node=3))
    p.set_local_agents({"other"})
    assert p.due({}, hub_online=True, now=0.0) == []


def test_a_failure_waits_one_interval():
    p = _planner(interval=2.0)
    p.on_version(_info(version=1))
    [_] = p.due({}, hub_online=True, now=0.0)
    p.failed("main")
    assert p.stats.failed == 1 and not p.in_flight("main")
    assert p.due({}, hub_online=True, now=1.9) == []
    assert len(p.due({}, hub_online=True, now=2.0)) == 1


def test_without_the_hub_direct_sources_are_polled_and_relay_only_ones_are_not():
    p = _planner(interval=1.0)
    p.set_referenced({"main", "far"})
    p.on_version(_info("main", 4))
    p.on_version(_info("far", 4, node=3, addresses=(), relay_only=True))
    jobs = p.due({"main": 4, "far": 4}, hub_online=False, now=0.0)
    assert [(j.agent_id, j.mode, j.since_version) for j in jobs] == [("main", "poll", 4)]
    assert p.finished("main", None, cached=4) is False           # GetLatest: not modified
    assert p.stats.not_modified == 1
    assert [j.agent_id for j in p.due({"main": 4}, hub_online=False, now=1.0)] == ["main"]
    assert p.finished("main", 6, cached=4) is True               # newer than announced: taken


def test_a_version_at_or_below_the_cached_one_is_redundant():
    p = _planner()
    p.on_version(_info(version=3))
    [_] = p.due({}, hub_online=True, now=0.0)
    assert p.finished("main", 3, cached=3) is False
    assert p.stats.redundant == 1 and p.stats.fetched == 0
    assert p.stats.as_counters() == {"weight_fetches": 0, "weight_fetch_failed": 0,
                                     "weight_fetch_not_modified": 0, "weight_fetch_redundant": 1}


def test_announcements_never_go_back_but_a_new_address_replaces_the_source():
    p = _planner()
    assert p.on_version(_info(version=5)) is True
    assert p.on_version(_info(version=4)) is False
    assert p.on_version(_info(version=5)) is False                # the same version from the same place
    assert p.on_version(_info(version=5, addresses=("10.0.0.9:47102",))) is True
    assert p.known("main").addresses == ("10.0.0.9:47102",)
    p.forget("main")
    assert p.known("main") is None


def test_the_interval_must_be_positive():
    with pytest.raises(ValueError):
        FetchPlanner(node=1, sync_interval_sec=0)
