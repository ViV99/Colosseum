"""Fixed players in the rollout loop, the launcher and the coordinator (SP3 T1.4, spec blocks 1 and 3)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import NetworkConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import build_model, build_network, env_spec
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, Lineup, SeatAssignment, state_dict_to_numpy
from colosseum.launcher import _resolve_lineups, setup_run
from colosseum.players.registry import BotSpec, FixedPlayers, FrozenSpec
from colosseum.worker.match_runner import ScriptedPlayer
from game_harness import GameFactory, lineup, make_loop, run_steps
from game_helpers import (
    RecordingBot,
    Tick,
    TickGame,
    agent_role_of,
    frozen_agent,
    make_coordinator,
    make_test_config,
    make_test_model,
    scripted_agent,
)

ROLE2 = TickGame([Tick(acting={0})], 2).spec.roles["player"]
WIDE = NetworkConfig.model_validate({"model_class": "game_helpers.GameTestModel", "kwargs": {"core": "none",
                                                                                             "hidden": 32}})


def _episode(length):
    return [Tick(acting={0, 1}, rewards={0: 1.0, 1: 1.0}) for _ in range(length)] + [
        Tick(over=True, rewards={0: 1.0, 1: -1.0})]


def _wide_frozen() -> FrozenSpec:
    model = build_network(WIDE, ROLE2)
    return FrozenSpec(agent_id="old", roles=("player",), networks=WIDE.model_dump(mode="json", by_alias=True),
                      model_state=state_dict_to_numpy(model.state_dict()), source="in-test")


def sd(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32)}


def test_the_rollout_loop_serves_fixed_players_that_never_collect():
    RecordingBot.instances.clear()
    fixed = FixedPlayers(bots={"bot": BotSpec("game_helpers.RecordingBot", {})}, frozen={"old": _wide_frozen()},
                         roles={"bot": ("player",), "old": ("player",)})
    loop, col = make_loop(
        GameFactory((_episode(3), 2)), {"a": lambda: make_test_model(ROLE2)},
        [lineup("2p", "a", SeatAssignment("bot", FIXED_NETWORK_ID, False)),
         lineup("2p", SeatAssignment("old", FIXED_NETWORK_ID, False), "a")],
        fixed_players=fixed,
    )
    try:
        assert isinstance(loop.get("bot", FIXED_NETWORK_ID), ScriptedPlayer)
        old = loop.get("old", FIXED_NETWORK_ID)
        assert not old.training
        assert sum(p.numel() for p in old.parameters()) > sum(p.numel() for p in loop.get("a", "latest").parameters())
        assert loop.get("bot", LATEST_NETWORK_ID) is None and loop.get("nobody", FIXED_NETWORK_ID) is None
        run_steps(loop, 12)                                   # 4 episodes per env
    finally:
        loop.close()
    assert col.chunks and {c.agent_id for c in col.chunks} == {"a"}
    assert {(s.agent_id, s.network_id) for r in col.results for s in r.seats} == {
        ("a", "latest"), ("bot", "fixed"), ("old", "fixed")}
    assert loop.stats["recorded_transitions"] == 3 * 4 * 2   # only the latest seats of "a" record
    assert len(RecordingBot.instances) == 1 and len(RecordingBot.instances[0].acts) == 3 * 4
    RecordingBot.instances.clear()


def test_resolve_lineups_keeps_fixed_seats_and_sources(tmp_path):
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.checkpoint_manager.save("agent_0", 10, sd(10))
    anchored = Lineup("2p", [SeatAssignment("agent_0", source="owner"),
                             SeatAssignment("rnd", FIXED_NETWORK_ID, False, source="anchors")])
    snapshot = Lineup("2p", [SeatAssignment("agent_0", source="owner"),
                             SeatAssignment("agent_0", "ckpt_v10", False, source="snapshots")])
    new_ckpts, resolved = _resolve_lineups([anchored, snapshot], coord, ["agent_0"])
    assert resolved == [anchored, snapshot]
    assert list(new_ckpts["agent_0"]) == ["ckpt_v10"] and "rnd" not in new_ckpts
    missing = Lineup("2p", [SeatAssignment("agent_0", source="owner"),
                            SeatAssignment("agent_0", "ckpt_v99", False, source="snapshots")])
    _new, (fallback,) = _resolve_lineups([missing], coord, ["agent_0"])
    assert fallback.seats[1] == SeatAssignment("agent_0", LATEST_NETWORK_ID, True, "snapshots")


def test_the_coordinator_rotates_owners_over_trainable_agents_in_config_order(tmp_path):
    cfg = make_test_config("turns", agents={"rnd": scripted_agent(), "a": {}, "b": {}},
                           matchmaking={"mode": "self_play", "latest_prob": 1.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    assert coord.player_roles == {"rnd": ["player"], "a": ["player"], "b": ["player"]}
    assert coord.agent_roles == {"a": ["player"], "b": ["player"]}
    lineups = coord.generate_lineups(4, env_offset=0)
    assert [next(s.agent_id for s in lu.seats if s.collect) for lu in lineups] == ["a", "b", "a", "b"]
    assert all(s.agent_id != "rnd" for lu in lineups for s in lu.seats)   # SP2 knobs: no anchors
    with pytest.raises(ConfigError, match=r"no roles resolved for agents \['rnd'\]"):
        Coordinator(cfg, env_spec(cfg), {"a": ["player"], "b": ["player"]}, tmp_path / "ckpt2")
    assert not hasattr(coord, "agent_pool")
    assert importlib.util.find_spec("colosseum.coordinator.agent_pool") is None


def test_setup_run_resolves_every_player_and_loads_fixed_players(tmp_path):
    base = make_test_config("turns")
    _roles, role = agent_role_of(base, "agent_0")
    pt = tmp_path / "old.pt"
    torch.save(build_model(base.get_agent_config("agent_0"), role).state_dict(), pt)
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent(), "old": frozen_agent(pt)})
    setup = setup_run(cfg, validate=False)
    assert setup.player_roles == {"agent_0": ["player"], "rnd": ["player"], "old": ["player"]}
    assert setup.agent_roles == {"agent_0": ["player"]} and list(setup.agent_configs) == ["agent_0"]
    assert setup.fixed.bots == {"rnd": BotSpec("colosseum.players.RandomBot", {})}
    assert list(setup.fixed.frozen) == ["old"] and setup.fixed.roles == {"rnd": ("player",), "old": ("player",)}
    bad = make_test_config("turns", agents={"agent_0": {}, "old": frozen_agent(tmp_path / "missing.pt")})
    with pytest.raises(ConfigError, match="expected a checkpoint dir"):
        setup_run(bad, validate=False)


def test_worker_main_passes_fixed_players_to_the_rollout_worker(monkeypatch):
    import colosseum.worker.rollout_worker as rollout_worker
    from colosseum.launcher import _worker_main

    recorded: dict = {}
    monkeypatch.setattr(rollout_worker, "rollout_worker_process", lambda **kwargs: recorded.update(kwargs))
    cfg = make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})
    setup = setup_run(cfg, validate=False)
    lineups = [Lineup("2p", [SeatAssignment("agent_0"), SeatAssignment("rnd", FIXED_NETWORK_ID, False)])
               for _ in range(cfg.rollout.envs_per_worker)]
    _worker_main(worker_id=0, config=cfg, agent_ids=["agent_0"], agent_roles=setup.agent_roles,
                 agent_configs=setup.agent_configs, role_specs=setup.role_specs, trajectory_queues={},
                 weight_queues={}, stop_event=None, lineups=lineups, fixed_players=setup.fixed)
    assert recorded["fixed_players"] is setup.fixed and recorded["lineups"] == lineups
