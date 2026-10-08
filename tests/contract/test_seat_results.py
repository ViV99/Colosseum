"""Per-seat match results from the real worker loop to the coordinator (T3.4).

4-seat FFA where agent "a" holds seats 0 and 2: no seat may be lost or
overwritten, neither in the worker's MatchResult nor in the coordinator.
"""
from pathlib import Path

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from dataflow_helpers import EnvFactory, FFA4Env, ProbeModel, make_loop, two_seat_result

REPO = Path(__file__).resolve().parents[2]


def _coordinator(tmp_path, agents):
    data = load_config(REPO / "configs/examples/tic_tac_toe.yaml").model_dump()
    data["checkpoint"]["dir"] = str(tmp_path / "ckpt")
    data["metrics"]["use_wandb"] = False
    coord = Coordinator(ColosseumConfig(**data))
    for aid in agents:
        coord.agent_pool.register_trainable(aid)
    return coord


def test_worker_reports_every_seat_of_a_four_seat_ffa():
    loop, col = make_loop(EnvFactory(FFA4Env), ProbeModel, agent_ids=("a", "b", "c"), num_envs=1,
                          slot_agent_map=[["a", "b", "a", "c"]],
                          collect_mask=[[True, True, True, True]])
    loop.step()
    loop.step()
    assert len(col.results) == 1
    result = col.results[0]
    assert result.match_id == "w0_e0_ep0" and result.episode_length == 2
    assert [(s.seat, s.agent_id, s.network_id, s.rank) for s in result.seats] == [
        (0, "a", "latest", 1), (1, "b", "latest", 2), (2, "a", "latest", 3), (3, "c", "latest", 4)]
    assert [s.outcome for s in result.seats] == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])
    assert [s.reward for s in result.seats] == pytest.approx([3.0, 2.0, 1.0, 0.0])


def test_coordinator_consumes_seats_without_collisions(tmp_path):
    loop, col = make_loop(EnvFactory(FFA4Env), ProbeModel, agent_ids=("a", "b", "c"), num_envs=1,
                          slot_agent_map=[["a", "b", "a", "c"]],
                          collect_mask=[[True, True, True, True]])
    loop.step()
    loop.step()
    result = col.results[0]
    by_agent = Coordinator._seat_outcomes_by_agent(result)
    assert set(by_agent) == {"a", "b", "c"}
    assert by_agent["a"] == pytest.approx([1.0, 1 / 3])
    assert by_agent["b"] == pytest.approx([2 / 3])
    assert by_agent["c"] == pytest.approx([0.0])
    coord = _coordinator(tmp_path, ("a", "b", "c"))
    coord.report_match_result(result)
    assert len(coord.match_results[-1].seats) == 4
    assert coord.elo.get("a") > coord.elo.get("c")   # a averages 2/3 over its two seats
    assert coord.elo.get("b") > coord.elo.get("c")


def test_same_agent_seats_carry_no_cross_agent_signal(tmp_path):
    coord = _coordinator(tmp_path, ("a",))
    coord.report_match_result(two_seat_result("a", 1.0, "a", 0.0, network_b="ckpt_v1"))
    assert coord.win_rates.get_win_rate("a", "a") == 0.5
    assert coord.elo.get("a") == coord.elo.initial_rating
