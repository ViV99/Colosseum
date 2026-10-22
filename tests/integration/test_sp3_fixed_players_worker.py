"""A spawned worker process builds and seats scripted and frozen players from FixedPlayers (SP3 T1.4); since SP5
(T2.5 b) the node's worker: bots from the node's spawn arguments, a frozen agent's weights from the hub's blob (against
``node_kit.FakeHub``, with the trainable agent's learner on the node)."""
from __future__ import annotations

import torch

from colosseum.core.registry import build_model
from colosseum.core.types import FIXED_NETWORK_ID, Lineup, SeatAssignment
from game_helpers import agent_role_of, frozen_agent, make_test_config, scripted_agent
from node_kit import intake_drops, intake_reports, train_on_node, train_steps, wait_for, worker_counters

WIDE_KWARGS = {"core": "none", "hidden": 32}


def test_a_spawned_worker_seats_scripted_and_frozen_players_that_never_collect(tmp_path):
    base = make_test_config("turns", networks={"kwargs": WIDE_KWARGS})
    _roles, role = agent_role_of(base, "agent_0")
    pt = tmp_path / "wide.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    config = make_test_config("turns", rollout={"envs_per_worker": 2, "chunk_length": 4},
                              learner={"batch_chunks": 1, "device": "cpu"}, agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(pt, networks={"kwargs": WIDE_KWARGS})})
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("rnd", FIXED_NETWORK_ID, False)]),
               Lineup("2p", [SeatAssignment("wide", FIXED_NETWORK_ID, False), SeatAssignment("agent_0")])]
    hub, node = train_on_node(tmp_path, config, learners=["agent_0"], workers=1, lineup=lambda w, e: lineups[e],
                              frozen={"wide": pt})
    try:
        wait_for(lambda: len(hub.results()) >= 4 and worker_counters(hub).get("chunks_sent", 0) >= 2
                 and train_steps(hub, "agent_0") >= 1, timeout=120)
        got = hub.results()
        counters = worker_counters(hub)
        drops = intake_drops(hub, "agent_0", after=len(intake_reports(hub, "agent_0")))   # a report after the chunks
    finally:
        node.stop()
        hub.stop()
    assert {(s.agent_id, s.network_id) for r in got for s in r.seats} == {
        ("agent_0", "latest"), ("rnd", FIXED_NETWORK_ID), ("wide", FIXED_NETWORK_ID)}
    # every chunk is agent_0's: none went without a learner (fixed seats never record), none dropped by its learner
    assert counters.get("dropped_chunks_no_learner", 0) == 0 and not drops
