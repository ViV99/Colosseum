"""Per-seat match results from the real worker loop to the coordinator (T3.4).

4-seat FFA where agent "a" holds seats 0 and 2: no seat may be lost or
overwritten, neither in the worker's MatchResult nor in the coordinator.
"""
from pathlib import Path

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.errors import EnvContractError
from colosseum.core.outcomes import outcomes_from_rewards
from dataflow_helpers import EnvFactory, FFA4Env, ProbeModel, make_loop, two_seat_result

REPO = Path(__file__).resolve().parents[2]


def _coordinator(tmp_path, agents):
    data = load_config(REPO / "configs/examples/tic_tac_toe.yaml").model_dump()
    data["metrics"]["use_wandb"] = False
    coord = Coordinator(ColosseumConfig(**data), checkpoint_dir=tmp_path / "ckpt")
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
    coord = _coordinator(tmp_path, ("a", "b", "c"))
    coord.report_match_result(result)
    assert len(coord.match_results[-1].seats) == 4
    # Every cross-agent seat pair counts: a holds seats 0 (1st) and 2 (3rd).
    wr = coord.win_rates
    assert (wr.games("a", "b"), wr.games("a", "c"), wr.games("b", "c")) == (2, 2, 1)
    assert wr.get_win_rate("a", "b") == 0.5   # a0 > b1, a2 < b1
    assert wr.get_win_rate("a", "c") == 1.0
    assert wr.get_win_rate("b", "c") == 1.0
    assert coord.elo.get("a") > coord.elo.get("c")
    assert coord.elo.get("b") > coord.elo.get("c")


def test_same_agent_seats_carry_no_cross_agent_signal(tmp_path):
    coord = _coordinator(tmp_path, ("a",))
    coord.report_match_result(two_seat_result("a", 1.0, "a", 0.0, network_b="ckpt_v1"))
    assert coord.win_rates.get_win_rate("a", "a") == 0.5
    assert coord.elo.get("a") == coord.elo.initial_rating
    assert coord.past_win_rate.get("a") == 1.0


def _ffa_result(terminal_info):
    loop, col = make_loop(EnvFactory(FFA4Env, terminal_info=terminal_info), ProbeModel,
                          agent_ids=("a", "b", "c", "d"), num_envs=1,
                          slot_agent_map=[["a", "b", "c", "d"]],
                          collect_mask=[[True, True, True, True]])
    loop.step()
    loop.step()
    assert len(col.results) == 1
    return col.results[0]


def test_worker_falls_back_to_reward_outcomes_without_env_signal():
    result = _ffa_result(lambda p: {})
    assert [s.rank for s in result.seats] == [None] * 4
    assert [s.reward for s in result.seats] == pytest.approx([3.0, 2.0, 1.0, 0.0])
    assert [s.outcome for s in result.seats] == outcomes_from_rewards([3.0, 2.0, 1.0, 0.0])
    assert [s.outcome for s in result.seats] == [1.0, 0.0, 0.0, 0.0]


def test_worker_keeps_fractional_ranks():
    ranks = [1, 2.5, 2.5, 4]
    result = _ffa_result(lambda p: {"rank": ranks[p]})
    assert [s.rank for s in result.seats] == [1.0, 2.5, 2.5, 4.0]
    assert [s.outcome for s in result.seats] == pytest.approx([1.0, 0.5, 0.5, 0.0])


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan")])
def test_worker_rejects_env_outcome_outside_unit_interval(bad):
    with pytest.raises(EnvContractError, match="outcome"):
        _ffa_result(lambda p: {"outcome": bad if p == 0 else 0.0})


def test_worker_rejects_non_finite_rank():
    with pytest.raises(EnvContractError, match="rank"):
        _ffa_result(lambda p: {"rank": float("inf") if p == 0 else p + 1})
