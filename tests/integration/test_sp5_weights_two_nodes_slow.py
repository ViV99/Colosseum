"""Criterion 4 across two machines (SP5 T3.1, spec 4.5, 4.6): a second node follows the learner's versions with one
network download per version, counted where it happens and summed by the hub.

``cluster_kit`` on localhost: node a runs the learner and one worker (it trains), node b one worker that plays the
learner's latest weights (without collecting until T3.2, amendment A11). 3 loaded processes: slow."""
from __future__ import annotations

import pytest

from cluster_kit import cluster, cluster_config, hub_counter, journal_rows

pytestmark = pytest.mark.slow


def test_the_second_node_follows_the_learners_versions_with_one_download_each(tmp_path):
    config = cluster_config(tmp_path, {"rollout.weight_sync_interval_sec": 1.0, "training.total_timesteps": 60000})
    nodes = [dict(name="a", cpu_learners=1, workers=1), dict(name="b", workers=1, no_learners=True)]
    with cluster(tmp_path, config, nodes, learner_first="agent_0") as (hub, (a, b)):
        hub.thread.join(240)                                    # the hub stops at the budget
        assert not hub.thread.is_alive() and hub.exit_codes == [0]
    # The nodes stopped (cluster): their counters are final. A fetch that ended during b's stop, after its last
    # heartbeat, is in b's count but not in the hub's total; the hub's total is what b's heartbeats carried.
    history = b.agent.fetch_history()["agent_0"]
    assert history == sorted(set(history)) and len(history) >= 3   # strictly increasing: no version twice
    b_counters, a_counters = b.agent.counters(), a.agent.counters()
    assert b_counters["weight_fetches"] == len(history) and b_counters["weight_fetch_redundant"] == 0
    assert a_counters.get("weight_fetches", 0) == 0             # its own learner is never fetched
    assert 3 <= hub_counter(hub, "weight_fetches") <= len(history)
    b_rows = [r for r in journal_rows(hub) if f"/n{b.agent.number}/" in r.uid]
    versions = {s.version_end for r in b_rows for s in r.seats if s.agent_id == "agent_0"}
    assert len(versions) >= 3                                   # b played with new weights
