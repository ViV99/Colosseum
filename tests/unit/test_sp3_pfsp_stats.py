"""PFSP statistics per player, fixed agents as rating entities, the anchor opponent type (SP3 T2.3, block 5)."""
from __future__ import annotations

import json

import gymnasium
import numpy as np
import pytest

from colosseum.coordinator.ratings import MemberPair, RatingBook, member_pairs
from colosseum.core.types import FIXED_NETWORK_ID, MatchResult, SeatResult, TeamResult
from colosseum.envs.game import GameSpec
from colosseum.league.pfsp import (
    DEFAULT_HALFLIFE_GAMES,
    PFSP_MIN_WEIGHT,
    PfspStats,
    pfsp_weight,
    player_name,
)
from colosseum.metrics.aggregator import OPPONENT_TYPES, EpisodeAggregator, opponent_type
from colosseum.metrics.hub import MetricsHub, wr_arena_over_layouts
from colosseum.metrics.jsonl import MetricsWriter
from game_helpers import make_coordinator, make_test_config, scripted_agent

SPEC = GameSpec.symmetric([2], gymnasium.spaces.Box(0.0, 1.0, (2,), np.float32), gymnasium.spaces.Discrete(3))


def seat(i, team, agent, network="latest", role="player") -> SeatResult:
    return SeatResult(seat=i, role=role, team=team, agent_id=agent, network_id=network, reward=0.0)


def duel(a, net_a, b, net_b, rank_a, rank_b, layout="2p") -> MatchResult:
    return MatchResult(match_id="m", layout=layout, outcome_kind="wdl",
                       seats=[seat(0, 0, a, net_a), seat(1, 1, b, net_b)],
                       teams=[TeamResult(0, float(rank_a), 0.0), TeamResult(1, float(rank_b), 0.0)], episode_length=3)


def step(weight, halflife=DEFAULT_HALFLIFE_GAMES) -> float:
    return 1.0 - 2.0 ** (-weight / halflife)


def test_pfsp_weights():
    assert pfsp_weight(0.25, "hard", 2.0) == pytest.approx(0.5625)
    assert pfsp_weight(0.25, "balanced", 7.0) == pytest.approx(0.1875)
    assert pfsp_weight(0.9, "uniform", 2.0) == 1.0
    assert pfsp_weight(1.0, "hard", 2.0) == PFSP_MIN_WEIGHT == 1e-6
    assert pfsp_weight(0.0, "balanced", 1.0) == PFSP_MIN_WEIGHT
    with pytest.raises(ValueError, match="weighting"):
        pfsp_weight(0.5, "steep", 1.0)


def test_the_ema_starts_at_the_prior_and_steps_by_the_halflife():
    stats = PfspStats({"a": 2.0})
    past = ("a", "ckpt_v3")
    assert stats.score("2p", "a", past) == 0.5 and stats.games("2p", "a", past) == 0.0
    stats.update(duel("a", "latest", "a", "ckpt_v3", 1, 2))     # a@latest beats its own snapshot
    assert stats.score("2p", "a", past) == pytest.approx(0.5 + step(1.0, 2.0) * 0.5)
    stats.update(duel("a", "ckpt_v3", "a", "latest", 1, 2))     # and loses from the other side
    x = 0.5 + step(1.0, 2.0) * 0.5
    assert stats.score("2p", "a", past) == pytest.approx(x + step(1.0, 2.0) * (0.0 - x))
    assert stats.games("2p", "a", past) == 2.0
    with pytest.raises(ValueError, match="halflife"):
        PfspStats({"a": 0.0})


def test_only_pairs_with_an_owner_at_latest_count_and_networks_are_told_apart():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES, "b": DEFAULT_HALFLIFE_GAMES})
    stats.update(duel("a", "latest", "b", "latest", 1, 2))           # both are owners
    stats.update(duel("a", "latest", "b", "ckpt_v4", 2, 1))          # a against b's snapshot
    stats.update(duel("a", "ckpt_v1", "b", "ckpt_v4", 1, 2))         # nobody at latest: ignored
    stats.update(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 1))  # a draw against an anchor
    stats.update(duel("a", "latest", "a", "latest", 1, 2))           # O@latest against O@latest: no pair
    snap = stats.snapshot()
    assert set(snap) == {"2p"} and set(snap["2p"]) == {"a", "b"}     # fixed agents are never owners
    assert set(snap["2p"]["a"]) == {"b@latest", "b@ckpt_v4", "rnd@fixed"}
    assert set(snap["2p"]["b"]) == {"a@latest"}
    assert snap["2p"]["a"]["rnd@fixed"] == {"score": pytest.approx(0.5), "games": 1.0}
    assert stats.score("2p", "a", ("b", "latest")) > 0.5 > stats.score("2p", "b", ("a", "latest"))
    assert stats.score("2p", "a", ("b", "ckpt_v4")) < 0.5
    assert stats.score("4p", "a", ("b", "latest")) == 0.5                # per layout
    assert player_name(("rnd", FIXED_NETWORK_ID)) == "rnd@fixed"


def test_team_pairs_split_their_weight_and_owners_outside_the_map_are_not_tracked():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES})
    two_v_two = MatchResult(match_id="m", layout="2v2", outcome_kind="wdl",
                            seats=[seat(0, 0, "a"), seat(1, 0, "a"),
                                   seat(2, 1, "b", "ckpt_v1"), seat(3, 1, "b", "ckpt_v1")],
                            teams=[TeamResult(0, 1.0, 0.0), TeamResult(1, 2.0, 0.0)], episode_length=3)
    stats.update(two_v_two)                                           # four member pairs of weight 1/4
    assert stats.games("2v2", "a", ("b", "ckpt_v1")) == pytest.approx(1.0)
    expected = 0.5
    for _ in range(4):
        expected += step(0.25) * (1.0 - expected)
    assert stats.score("2v2", "a", ("b", "ckpt_v1")) == pytest.approx(expected)
    stats.update(duel("b", "latest", "a", "ckpt_v2", 1, 2))           # b is not in the halflife map
    assert "2p" not in stats.snapshot()


def test_forget_drops_a_player_in_every_layout_and_for_every_owner():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES, "b": DEFAULT_HALFLIFE_GAMES})
    stats.update(duel("a", "latest", "b", "ckpt_v4", 1, 2))
    stats.update(duel("a", "latest", "b", "ckpt_v4", 1, 2, layout="4p"))
    stats.update(duel("a", "latest", "b", "latest", 1, 2))
    stats.forget(("b", "ckpt_v4"))
    assert stats.games("2p", "a", ("b", "ckpt_v4")) == 0.0 and stats.games("4p", "a", ("b", "ckpt_v4")) == 0.0
    assert stats.games("2p", "a", ("b", "latest")) == 1.0


def test_a_late_result_never_brings_a_forgotten_snapshot_back():
    stats = PfspStats({"a": DEFAULT_HALFLIFE_GAMES})
    stats.update(duel("a", "latest", "a", "ckpt_v10", 1, 2))
    stats.forget(("a", "ckpt_v10"))
    stats.update(duel("a", "latest", "a", "ckpt_v10", 1, 2))         # an episode in progress at the eviction
    stats.update(duel("a", "latest", "a", "ckpt_v20", 2, 1))
    assert "a@ckpt_v10" not in stats.snapshot()["2p"]["a"]
    assert stats.games("2p", "a", ("a", "ckpt_v10")) == 0.0
    assert stats.games("2p", "a", ("a", "ckpt_v20")) == 1.0


def test_late_results_after_evictions_keep_the_coordinator_table_small(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    state = {"w": np.zeros((2, 3), np.float32)}
    coord.checkpoint_manager.save("agent_0", 10, state)
    for version in (20, 30, 40):
        coord.checkpoint_manager.save("agent_0", version, state)       # evicts the previous snapshot
        coord.report_match_result(duel("agent_0", "latest", "agent_0", f"ckpt_v{version - 10}", 1, 2))
    assert coord.ratings_snapshot()["2p"].get("pfsp", {}).get("agent_0", {}) == {}


def test_member_pairs_carry_the_network_ids_of_both_sides():
    assert member_pairs(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2)) == [
        MemberPair("cross", "a", "rnd", 1.0, 1.0, "player", "player", "latest", "fixed")]
    assert member_pairs(duel("a", "ckpt_v3", "a", "latest", 1, 2)) == [
        MemberPair("past", "a", "a", 0.0, 1.0, "player", "player", "latest", "ckpt_v3")]


def test_scripted_and_frozen_agents_are_rating_entities():
    book = RatingBook(SPEC, ["a", "rnd"])
    book.update(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2))
    snap = book.snapshot()["2p"]
    assert snap["elo"]["a"] > 1200.0 > snap["elo"]["rnd"]
    assert snap["win_rates"]["a"]["rnd"] == 1.0 and snap["games"]["rnd"]["a"] == 1
    assert snap["wr_vs_past"]["rnd"] is None
    assert snap["role_win_rates"]["player"]["rnd"]["a"] == 0.0


def test_the_coordinator_rates_fixed_agents_and_publishes_the_pfsp_table(tmp_path):
    cfg = make_test_config("turns", agents={"a": {}, "rnd": scripted_agent()})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    coord.report_match_result(duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2))
    snap = coord.ratings_snapshot()["2p"]
    assert set(snap["elo"]) == {"a", "rnd"}
    assert snap["pfsp"] == {"a": {"rnd@fixed": {"score": pytest.approx(0.5 + step(1.0) * 0.5), "games": 1.0}}}
    assert coord.pfsp.score("2p", "a", ("rnd", FIXED_NETWORK_ID)) > 0.5
    json.dumps(coord.ratings_snapshot())                             # ratings.json stays JSON


def test_an_evicted_snapshot_is_forgotten_by_pfsp(tmp_path):
    cfg = make_test_config("turns", checkpoint={"keep_last": 1, "keep_every": 0})
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    state = {"w": np.zeros((2, 3), np.float32)}
    coord.checkpoint_manager.save("agent_0", 10, state)
    coord.report_match_result(duel("agent_0", "latest", "agent_0", "ckpt_v10", 1, 2))
    assert coord.pfsp.games("2p", "agent_0", ("agent_0", "ckpt_v10")) == 1.0
    coord.checkpoint_manager.save("agent_0", 20, state)               # evicts ckpt_v10
    assert coord.pfsp.games("2p", "agent_0", ("agent_0", "ckpt_v10")) == 0.0
    assert "agent_0@ckpt_v10" not in json.dumps(coord.ratings_snapshot())


def test_an_opponent_played_by_a_fixed_agent_is_an_anchor():
    assert OPPONENT_TYPES == ("latest", "past", "arena", "anchor")
    result = duel("a", "latest", "rnd", FIXED_NETWORK_ID, 1, 2)
    assert opponent_type(result, result.seats[0]) == "anchor"
    team = MatchResult(match_id="m", layout="2v2", outcome_kind="wdl",
                       seats=[seat(0, 0, "a"), seat(1, 0, "a"), seat(2, 1, "b"), seat(3, 1, "rnd", FIXED_NETWORK_ID)],
                       teams=[TeamResult(0, 1.0, 0.0), TeamResult(1, 2.0, 0.0)], episode_length=3)
    assert opponent_type(team, team.seats[0]) == "anchor" and opponent_type(team, team.seats[2]) == "arena"
    agg = EpisodeAggregator()
    agg.add(result)
    out = agg.flush()
    assert set(out) == {"a"}                                          # fixed seats are not aggregated
    assert out["a"]["wdl"] == {"latest": [0, 0, 0], "past": [0, 0, 0], "arena": [0, 0, 0], "anchor": [1, 0, 0]}


def test_wandb_rows_leave_out_the_pfsp_tables(tmp_path):
    class Rows:
        def __init__(self) -> None:
            self.rows: list[dict] = []

        def log_global(self, row, step):
            self.rows.append(row)

        def log_train(self, agent_id, values, step):
            pass

    rows = Rows()
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    hub = MetricsHub(writer=writer, ratings_path=tmp_path / "ratings.json", agent_ids=["a"], total_timesteps=10,
                     log_interval=1, console_interval_sec=0.0, wandb_logger=rows)
    ratings = {"2p": {"elo": {"a": 1210.0, "rnd": 1190.0}, "win_rates": {}, "games": {}, "wr_vs_past": {"a": None},
                      "past_games": {"a": 0}, "scores": {}, "cross_play": {}, "role_win_rates": {},
                      "pfsp": {"a": {"rnd@fixed": {"score": 0.6, "games": 1.0}}}}}
    try:
        hub.maybe_tick(env_steps=5, ratings=ratings, queue_depths={}, force=True)
    finally:
        writer.close()
    assert rows.rows[0]["ratings/2p/elo/rnd"] == 1190.0
    assert not any("/pfsp/" in key for key in rows.rows[0])
    written = json.loads((tmp_path / "ratings.json").read_text())
    assert written["layouts"]["2p"]["pfsp"]["a"]["rnd@fixed"]["games"] == 1.0


def test_the_console_arena_win_rate_leaves_out_anchors(tmp_path, caplog):
    """Ruling P11: ``wr_arena`` averages over trainable opponents only; anchor games stay in the ratings."""
    ratings = {"2p": {"win_rates": {"a": {"b": 1.0, "rnd": 0.0}}, "games": {"a": {"b": 1, "rnd": 3}}}}
    assert wr_arena_over_layouts(ratings, "a") == pytest.approx(0.25)
    assert wr_arena_over_layouts(ratings, "a", opponents=["a", "b"]) == 1.0
    assert wr_arena_over_layouts(ratings, "a", opponents=["a"]) is None
    writer = MetricsWriter(tmp_path / "metrics.jsonl")
    hub = MetricsHub(writer=writer, ratings_path=tmp_path / "ratings.json", agent_ids=["a", "b"],
                     total_timesteps=10, log_interval=1, console_interval_sec=0.0)
    try:
        with caplog.at_level("INFO"):
            hub.maybe_tick(env_steps=5, ratings=ratings, queue_depths={}, force=True)
    finally:
        writer.close()
    assert "wr_arena 1.00" in caplog.text
