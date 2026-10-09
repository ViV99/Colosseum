"""Coordinator v2: owner rotation, lineups, per-layout ratings, roles in checkpoint meta (T5.3)."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.errors import ConfigError
from colosseum.core.registry import env_spec
from colosseum.core.roles import agent_role_spec, resolve_agent_roles, role_signature
from colosseum.core.types import LATEST_NETWORK_ID, MatchResult, SeatResult, TeamResult
from game_helpers import make_coordinator, make_test_config


def sd(value: float = 0.0) -> dict[str, np.ndarray]:
    return {"w": np.full((2, 3), value, np.float32)}


def owners(coord: Coordinator, envs: int, offset: int = 0) -> list[str]:
    """The agent whose latest weights collect in each lineup (the owner always does)."""
    out = []
    for lineup in coord.generate_lineups(envs, env_offset=offset):
        collecting = {s.agent_id for s in lineup.seats if s.collect}
        assert len(collecting) == 1, lineup  # self-play: only the owner's agent collects
        out.append(collecting.pop())
    return out


def test_owner_rotates_with_global_env_index_and_round(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}, "c": {}},
                           matchmaking={"mode": "self_play", "latest_prob": 1.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    assert owners(coord, 4) == ["a", "b", "c", "a"]
    assert owners(coord, 2, offset=1) == ["b", "c"]
    coord.next_round()
    assert coord.refresh_round == 1
    assert owners(coord, 3) == ["b", "c", "a"]


def test_asymmetric_agents_both_own_envs_and_fill_each_others_teams(tmp_path):
    cfg = make_test_config("asymmetric", matchmaking={"mode": "self_play"})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    lineups = coord.generate_lineups(8, env_offset=0)
    for lineup in lineups:
        roles = coord.spec.layouts[lineup.layout]
        for seat, seat_spec in zip(lineup.seats, roles, strict=True):
            assert seat_spec.role in coord.agent_roles[seat.agent_id]
    assert {s.agent_id for lu in lineups for s in lu.seats} == {"hunter", "prey"}


def test_self_play_draws_saved_checkpoints_as_opponents(tmp_path):
    cfg = make_test_config("turns", matchmaking={"mode": "self_play", "latest_prob": 0.0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    assert all(s.network_id == LATEST_NETWORK_ID for lu in coord.generate_lineups(4, 0) for s in lu.seats)
    coord.checkpoint_manager.save("agent_0", 7, sd(7))
    networks = Counter(s.network_id for lu in coord.generate_lineups(20, 0) for s in lu.seats)
    assert networks["ckpt_v7"] == 20 and networks[LATEST_NETWORK_ID] == 20  # one owner seat per 2-seat match


def test_report_match_result_updates_the_layout_ratings(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    layout = next(iter(coord.spec.layouts))
    role = coord.spec.role_of(layout, 0)
    result = MatchResult(
        match_id="m", layout=layout, outcome_kind="wdl",
        seats=[SeatResult(0, role, 0, "a", "latest", 1.0), SeatResult(1, role, 1, "b", "latest", -1.0)],
        teams=[TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], episode_length=4,
    )
    coord.report_match_result(result)
    snap = coord.ratings_snapshot()
    assert snap[layout]["elo"]["a"] > snap[layout]["elo"]["b"]
    assert coord.ratings.win_rate(layout, "a", "b") == 1.0
    assert coord.match_results == [result]


def test_checkpoint_meta_records_roles_and_signature(tmp_path):
    cfg = make_test_config("asymmetric")
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    payload = {"agent_id": "hunter", "policy_version": 3, "final": True, "model_state": sd(3),
               "trainer_state_bytes": b"opt"}
    assert coord.save_checkpoint_payload(payload, meta_extra={"env_steps": 12}) == "ckpt_v3"
    meta = json.loads((tmp_path / "ckpt" / "hunter" / "ckpt_v3" / "meta.json").read_text())
    spec = env_spec(cfg)
    expected = role_signature(agent_role_spec(spec, resolve_agent_roles(cfg, spec)["hunter"]))
    assert meta["roles"] == ["hunter"] and meta["role_signature"] == expected == coord.role_signature("hunter")
    assert meta["final"] is True and meta["env_steps"] == 12
    assert coord.role_signature("prey") != coord.role_signature("hunter")
    assert (tmp_path / "ckpt" / "hunter" / "ckpt_v3" / "trainer_state.pt").read_bytes() == b"opt"


def test_trainer_state_is_dropped_without_save_optimizer(tmp_path):
    cfg = make_test_config("solo", checkpoint={"save_optimizer": False})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.save_checkpoint_payload({"agent_id": "agent_0", "policy_version": 1, "model_state": sd(1),
                                   "trainer_state_bytes": b"opt"})
    assert not (tmp_path / "ckpt" / "agent_0" / "ckpt_v1" / "trainer_state.pt").exists()


def test_missing_roles_for_an_agent_is_a_config_error(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "b": {}})
    spec = env_spec(cfg)
    with pytest.raises(ConfigError, match="no roles"):
        Coordinator(cfg, spec, {"a": list(spec.roles)}, tmp_path / "ckpt")
