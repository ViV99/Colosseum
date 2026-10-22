"""Chunks between machines with a real hub and two real nodes (SP5 T3.2, spec 4.7, 4.8): node b's worker collects for
the learner on node a; its frames leave on its route connection, go through node b's router in one mutual-TLS stream
to node a's data port and reach the learner over node a's own chunk connection, undecoded on the way. The learner
trains only on them (node a runs no worker).

``cluster_kit``: the hub and both nodes in threads of the test process (TLS hub port, mutual-TLS data ports); the
loaded processes are node a's learner and node b's worker (2). Part (a): the direct path; the hub relay is T3.2 (b)."""
from __future__ import annotations

from cluster_kit import cluster, cluster_config, hub_version, journal_rows, wait_until

NODES = [dict(name="a", cpu_learners=1, workers=0), dict(name="b", workers=1, no_learners=True)]


def _drops(counters: dict) -> dict:
    return {k: v for k, v in counters.items() if k.startswith("dropped_chunks.") and v}


def _trains_over_the_direct_path(tmp_path, overrides):
    config = cluster_config(tmp_path, {"agents.main": {}, "training.total_timesteps": 10**9, **overrides})
    with cluster(tmp_path, config, NODES, learner_first="main") as (hub, (a, b)):
        assert wait_until(lambda: (hub_version(hub, "main") or 0) >= 4, 180), "the learner did not train"
        # a stream's frames are counted sent when it ends (STREAM_ROTATE_SEC)
        assert wait_until(lambda: b.agent.counters().get("chunks_sent.direct", 0) > 0, 30), b.agent.counters()
        b_counters, a_counters = b.agent.counters(), a.agent.counters()
        assert b_counters.get("chunks_sent.direct", 0) > 0 and "chunks_sent.relay" not in b_counters
        assert b_counters.get("chunks_routed", 0) >= b_counters["chunks_sent.direct"]
        assert a_counters.get("chunks_delivered", 0) > 0
        assert _drops(a_counters) == {} and _drops(b_counters) == {}
        assert b.agent.paths() == {a.agent.number: "direct"}
        assert a.agent.paths() == {}                          # its own learner: no remote target
        assert len(b.agent.fetch_history()["main"]) >= 2      # b follows the versions it trains

        def played_new_weights() -> bool:
            rows = [r for r in journal_rows(hub) if f"/n{b.agent.number}/" in r.uid]
            return any(s.version_end >= 1 for r in rows for s in r.seats if s.agent_id == "main")
        assert wait_until(played_new_weights, 60)
        assert hub.stop() == 0


def test_a_worker_on_another_node_trains_the_learner_directly(tmp_path):
    _trains_over_the_direct_path(tmp_path, {})


def test_relay_off_a_reachable_learner_collects_directly(tmp_path):
    """Pre-flight ruling P-PF3-1: with ``distributed.relay: false`` the node probes its path before any frame, reports
    ``direct`` and is granted collecting seats."""
    _trains_over_the_direct_path(tmp_path, {"distributed.relay": False})
