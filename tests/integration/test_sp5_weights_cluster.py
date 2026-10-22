"""Weights across machines with a real hub and two real nodes (SP5 T3.1, spec 4.5, 4.6, 3.1-3.2): the hub announces the
learner node's versions with its data addresses, the second node downloads them once over mutual TLS and plays with
them; frozen and teacher blobs reach every node by id, an ``init`` blob only the node that runs the agent's learner.

``cluster_kit``: the hub and both nodes run in threads of the test process (TLS hub port, mutual-TLS data ports);
the loaded processes are node a's learner and node b's worker (2). Until T3.2 node b's ``main`` seats do not collect
(amendment A11), so the learner stays at its initial version 0."""
from __future__ import annotations

import json

from cluster_kit import cluster, cluster_config, hub_counter, journal_results_by_node, journal_rows, wait_until
from colosseum.node.weights import WeightCacheReader, blob_key, shm_dir
from game_helpers import write_ckpt_dir

ANCHORS_ONLY = {"matchmaking.opponents": {"latest": 0.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 1.0},
                "training.total_timesteps": 10**9}
NODES = [dict(name="a", cpu_learners=1, workers=0), dict(name="b", workers=1, no_learners=True)]


def _source_dir(tmp_path):
    """A checkpoint dir of the tic-tac-toe model (its networks in ``meta.json``)."""
    base = cluster_config(tmp_path, {"agents.main": {}}, name="base")
    path = write_ckpt_dir(tmp_path / "source", base, "main", policy_version=3)
    return path, json.loads((path / "meta.json").read_text())["networks"]


def _cache(node):
    return WeightCacheReader(shm_dir(), node.agent.cache_prefix)


def _blob(hub, kind, agent):
    [fb] = [fb for fb in hub.hub._fixed_blobs if fb.kind == kind and fb.agent_id == agent]
    return blob_key(kind, agent, fb.blob_id)


def test_a_second_node_plays_with_the_learner_nodes_weights_and_the_frozen_blob(tmp_path):
    path, networks = _source_dir(tmp_path)
    config = cluster_config(tmp_path, {"agents.main": {}, "agents.frozen": {"kind": "frozen", "path": str(path)},
                                       **ANCHORS_ONLY})
    with cluster(tmp_path, config, NODES, learner_first="main") as (hub, (a, b)):
        assert wait_until(lambda: journal_results_by_node(hub).get(b.agent.number, 0) >= 20, 120)
        assert b.agent.fetch_history() == {"main": [0]}          # the learner's initial weights, downloaded once
        frozen = _cache(b).get(_blob(hub, "fixed", "frozen"))
        assert frozen is not None and frozen.meta.architecture == networks
        assert wait_until(lambda: hub_counter(hub, "weight_fetches") == 1, 30)
        assert a.agent.counters()["data_port.latest_served"] == 1
        assert a.agent.counters()["weight_fetches"] == 0         # its own learner is never fetched
        rows = [r for r in journal_rows(hub) if f"/n{b.agent.number}/" in r.uid]
        assert rows and all(any(s.agent_id == "frozen" and s.network_id == "fixed" for s in r.seats) for r in rows)
        assert all(s.version_end == 0 for r in rows for s in r.seats if s.agent_id == "main")
        assert hub.stop() == 0


def test_teacher_and_init_blobs_reach_a_second_node(tmp_path):
    path, networks = _source_dir(tmp_path)
    config = cluster_config(tmp_path, {"agents.main": {"kickstart": {"teacher": str(path)},
                                                       "init": {"from": str(path)}},
                                       "agents.random": {"kind": "scripted", "class": "colosseum.players.RandomBot"},
                                       **ANCHORS_ONLY})
    with cluster(tmp_path, config, NODES, learner_first="main") as (hub, (a, b)):
        assert wait_until(lambda: journal_results_by_node(hub).get(b.agent.number, 0) >= 5, 120)
        teacher, init = _blob(hub, "teacher", "main"), _blob(hub, "init", "main")
        cached = _cache(b).get(teacher)
        assert cached is not None and cached.meta.kind == "teacher" and cached.meta.architecture == networks
        assert init in b.agent._run.fixed_keys                    # listed in b's Welcome.fixed ...
        assert not _cache(b).has(init)                            # ... but fetched only by the learner's node
        assert _cache(a).has(init) and _cache(a).has(teacher)
        assert hub.stop() == 0
