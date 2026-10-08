"""Owner rotation over all trainable agents, N-player arenas, seat shuffling (T5.1)."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from colosseum.coordinator.coordinator import Coordinator
from colosseum.core.config import ColosseumConfig

TTT = "examples.tic_tac_toe"


def make_config(agents, *, phase="self_play", num_players=2, self_play_ratio=0.5,
                latest_prob=0.5, shuffle_seats=True, seed=0) -> ColosseumConfig:
    return ColosseumConfig.model_validate({
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": num_players},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "training": {"phase": phase, "seed": seed},
        "self_play": {"self_play_ratio": self_play_ratio, "latest_prob": latest_prob,
                      "shuffle_seats": shuffle_seats},
        "agents": {a: {} for a in agents},
    })


def make_coordinator(tmp_path, agents, **kw) -> Coordinator:
    return Coordinator(make_config(agents, **kw), checkpoint_dir=tmp_path / "ckpt")


def run_rounds(coord, *, workers, envs_per_worker, rounds):
    """Generate matches the way the launcher does: per worker, then advance the round."""
    matches = []
    for _ in range(rounds):
        for w in range(workers):
            matches += coord.generate_match_configs(envs_per_worker, env_offset=w * envs_per_worker)
        coord.next_round()
    return matches


def fake_checkpoints(monkeypatch, coord, ids=("ckpt_v10",)):
    monkeypatch.setattr(
        coord.checkpoint_manager, "list_checkpoints",
        lambda agent_id: [SimpleNamespace(checkpoint_id=c, agent_id=agent_id) for c in ids],
    )


def test_self_play_two_agents_both_get_collecting_envs(tmp_path):
    coord = make_coordinator(tmp_path, ["alpha", "beta"])
    matches = run_rounds(coord, workers=2, envs_per_worker=2, rounds=1)
    agents_per_match = [{s.agent_id for s in m.player_slots} for m in matches]
    assert all(len(a) == 1 for a in agents_per_match), "self-play must not mix agents"
    assert Counter(next(iter(a)) for a in agents_per_match) == {"alpha": 2, "beta": 2}
    collecting = Counter(s.agent_id for m in matches for s in m.player_slots if s.collect_trajectories)
    assert collecting["alpha"] > 0 and collecting["beta"] > 0


def test_owner_rotates_with_global_env_index_and_round(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b", "c"], shuffle_seats=False)

    def owner(g):
        return coord.generate_match_configs(1, env_offset=g)[0].player_slots[0].agent_id

    assert [owner(g) for g in range(4)] == ["a", "b", "c", "a"]
    coord.next_round()
    assert coord.refresh_round == 1
    assert [owner(g) for g in range(3)] == ["b", "c", "a"]


def test_league_three_agents_all_pairs_meet(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b", "c"], phase="league", self_play_ratio=0.0)
    matches = run_rounds(coord, workers=2, envs_per_worker=4, rounds=20)
    pairs = Counter(tuple(sorted({s.agent_id for s in m.player_slots})) for m in matches)
    assert {("a", "b"), ("a", "c"), ("b", "c")} <= set(pairs), pairs
    for m in matches:
        assert all(s.collect_trajectories and s.checkpoint_id is None for s in m.player_slots)


@pytest.mark.parametrize("phase,agents,ratio", [
    ("league", ["a", "b", "c"], 0.3),
    ("self_play", ["a", "b"], 0.5),
])
def test_seat_distribution_balanced_within_5_percent(tmp_path, monkeypatch, phase, agents, ratio):
    coord = make_coordinator(tmp_path, agents, phase=phase, self_play_ratio=ratio, seed=1)
    fake_checkpoints(monkeypatch, coord)
    matches = run_rounds(coord, workers=2, envs_per_worker=8, rounds=400)
    for agent in agents:
        seats = Counter(i for m in matches for i, s in enumerate(m.player_slots)
                        if s.agent_id == agent and s.collect_trajectories)
        total = sum(seats.values())
        assert total > 1000
        assert abs(seats[0] / total - 0.5) <= 0.05, (agent, seats)


def test_checkpoint_opponents_spread_over_seats(tmp_path, monkeypatch):
    coord = make_coordinator(tmp_path, ["a"], latest_prob=0.0, seed=2)
    fake_checkpoints(monkeypatch, coord)
    matches = run_rounds(coord, workers=1, envs_per_worker=8, rounds=200)
    ckpt_seats = Counter(i for m in matches for i, s in enumerate(m.player_slots) if s.checkpoint_id is not None)
    assert sum(ckpt_seats.values()) == len(matches)  # latest_prob=0: one checkpoint seat per match
    assert abs(ckpt_seats[0] / len(matches) - 0.5) <= 0.05
    for m in matches:
        assert all(not s.collect_trajectories for s in m.player_slots if s.checkpoint_id is not None)


def test_four_player_arena_has_four_slots(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b", "c"], phase="league", num_players=4, self_play_ratio=0.0)
    matches = run_rounds(coord, workers=1, envs_per_worker=6, rounds=5)
    for m in matches:
        assert len(m.player_slots) == 4
        assert all(s.collect_trajectories and s.checkpoint_id is None for s in m.player_slots)
    assert any(len({s.agent_id for s in m.player_slots}) == 3 for m in matches)


def test_shuffle_disabled_keeps_owner_in_seat_zero(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"], phase="league", self_play_ratio=0.0, shuffle_seats=False)
    matches = coord.generate_match_configs(4, env_offset=0)
    assert [m.player_slots[0].agent_id for m in matches] == ["a", "b", "a", "b"]


def test_pfsp_prefers_hard_opponents(tmp_path, monkeypatch):
    coord = make_coordinator(tmp_path, ["a0", "easy", "hard"], phase="league",
                             self_play_ratio=0.0, shuffle_seats=False)
    wr = {"easy": 0.9, "hard": 0.1}
    monkeypatch.setattr(coord.win_rates, "get_win_rate",
                        lambda a, b: wr.get(b, 0.5) if a == "a0" else 0.5)
    picks = Counter()
    for _ in range(600):
        match = coord.generate_match_configs(1, env_offset=0)[0]  # env 0, round 0 -> owner a0
        picks[match.player_slots[1].agent_id] += 1
    assert picks["hard"] > 3 * picks["easy"], picks


def test_single_player_env_gets_solo_matches(tmp_path):
    coord = make_coordinator(tmp_path, ["a", "b"], phase="league", num_players=1, self_play_ratio=0.0)
    matches = coord.generate_match_configs(4, env_offset=0)
    assert all(len(m.player_slots) == 1 and m.player_slots[0].collect_trajectories for m in matches)
