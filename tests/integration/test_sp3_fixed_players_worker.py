"""A spawned worker process builds and seats scripted and frozen players from FixedPlayers (SP3 T1.4)."""
from __future__ import annotations

import multiprocessing as mp

import torch

from colosseum.core.registry import build_model
from colosseum.core.types import FIXED_NETWORK_ID, Lineup, SeatAssignment, TrajectoryChunk
from colosseum.launcher import _worker_target, setup_run
from game_helpers import agent_role_of, frozen_agent, make_test_config, scripted_agent

WIDE_KWARGS = {"core": "none", "hidden": 32}


def test_a_spawned_worker_seats_scripted_and_frozen_players_that_never_collect(tmp_path):
    base = make_test_config("turns", networks={"kwargs": WIDE_KWARGS})
    _roles, role = agent_role_of(base, "agent_0")
    pt = tmp_path / "wide.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    config = make_test_config("turns", rollout={"envs_per_worker": 2, "chunk_length": 4}, agents={
        "agent_0": {}, "rnd": scripted_agent(), "wide": frozen_agent(pt, networks={"kwargs": WIDE_KWARGS})})
    setup = setup_run(config, validate=False)
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("rnd", FIXED_NETWORK_ID, False)]),
               Lineup("2p", [SeatAssignment("wide", FIXED_NETWORK_ID, False), SeatAssignment("agent_0")])]
    ctx = mp.get_context("spawn")
    trajectories, weights, results, stop = ctx.Queue(maxsize=64), ctx.Queue(maxsize=1), ctx.Queue(maxsize=100), \
        ctx.Event()
    proc = ctx.Process(target=_worker_target, daemon=True, kwargs=dict(
        worker_id=0, config=config, agent_ids=["agent_0"], agent_roles=setup.agent_roles,
        agent_configs=setup.agent_configs, role_specs=setup.role_specs,
        trajectory_queues={"agent_0": trajectories}, weight_queues={"agent_0": weights}, stop_event=stop,
        lineups=lineups, results_queue=results, fixed_players=setup.fixed,
    ))
    proc.start()
    try:
        got = [results.get(timeout=60) for _ in range(4)]
        chunks = [TrajectoryChunk.from_payload(trajectories.get(timeout=60)) for _ in range(2)]
    finally:
        stop.set()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    assert {(s.agent_id, s.network_id) for r in got for s in r.seats} == {
        ("agent_0", "latest"), ("rnd", FIXED_NETWORK_ID), ("wide", FIXED_NETWORK_ID)}
    assert all(c.agent_id == "agent_0" for c in chunks)
