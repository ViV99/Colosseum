"""Team-pair sufficient statistics (SP4 T1.2, spec blocks 1-2): team keys with team index and roles, weights
1/(T-1), outcomes from ranks, canonical order, self-play excluded from games, score layouts, copy / merge, and a
thread-safety smoke test."""
from __future__ import annotations

import threading

import pytest

from colosseum.arena.comparisons import (
    ComparisonTable,
    PairStat,
    ScoreStat,
    TeamKey,
    coefficients,
    match_pairs,
    match_score,
    team_keys,
)
from game_helpers import make_match_result


def test_team_keys_carry_the_team_index_entities_and_roles():
    result = make_match_result([(0, "main", "latest", "hunter", 1250), (1, "main", "ckpt_v700", "prey"),
                                (1, "random", "fixed", "prey")], layout="1v2")
    keys = team_keys(result, [1200, 700, None])
    assert keys == {0: TeamKey(0, (("main@v1200", "hunter"),)),
                    1: TeamKey(1, (("main@v700", "prey"), ("random", "prey")))}
    assert TeamKey.decode(keys[1].encode()) == keys[1]
    with pytest.raises(ValueError, match="rating versions"):
        team_keys(result, [1])


def test_pairs_weights_and_outcomes_from_ranks():
    duel = make_match_result([(0, "a", "fixed"), (1, "b", "fixed")], [2.0, 1.0])
    ((a, b, w, y),) = match_pairs(duel, [None, None])
    assert (a.team, b.team, w, y) == (0, 1, 1.0, 0.0)
    draw = make_match_result([(0, "a", "fixed"), (1, "b", "fixed")], [1.0, 1.0])
    assert match_pairs(draw, [None, None])[0][3] == 0.5
    ffa = make_match_result([(t, f"p{t}", "fixed") for t in range(4)], [2.0, 1.0, 4.0, 2.0], layout="4p")
    pairs = match_pairs(ffa, [None] * 4)
    assert len(pairs) == 6 and all(w == pytest.approx(1 / 3) for *_, w, _ in pairs)
    outcome = {(a.team, b.team): y for a, b, _, y in pairs}
    assert outcome == {(0, 1): 0.0, (0, 2): 1.0, (0, 3): 0.5, (1, 2): 1.0, (1, 3): 1.0, (2, 3): 0.0}
    solo = make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[7.5], layout="coop")
    assert match_pairs(solo, [None, None]) == []
    key, score = match_score(solo, [None, None])
    assert score == 7.5 and key.entities() == ["a", "b"]


def test_coefficients_cancel_an_entity_on_both_sides():
    a = TeamKey.of(0, [("x", "p"), ("y", "p")])
    b = TeamKey.of(1, [("x", "p"), ("z", "p")])
    assert coefficients(a, b) == {"y": 0.5, "z": -0.5}
    assert coefficients(TeamKey.of(0, [("x", "p")]), TeamKey.of(1, [("x", "p")])) == {}


def test_the_table_stores_pairs_in_canonical_order_and_sums_them():
    table = ComparisonTable()
    a, b = TeamKey.of(0, [("x", "p")]), TeamKey.of(1, [("y", "p")])
    table.add("2p", "train", b, a, PairStat(3, 3.0, 1.0))           # given as (b, a): stored as (a, b)
    table.add("2p", "train", a, b, PairStat(2, 2.0, 2.0))
    table.add("2p", "pulse", a, b, PairStat(1, 1.0, 0.0))
    assert table.pairs("2p", ["train"]) == [(a, b, PairStat(5, 5.0, 4.0))]
    assert table.pairs("2p", ["train", "pulse"]) == [(a, b, PairStat(6, 6.0, 4.0))]
    assert table.pairs("2p", ["tournament"]) == [] and table.layouts() == ["2p"]
    assert table.sources("2p") == ["pulse", "train"]
    with pytest.raises(ValueError, match="team indices"):
        table.add("2p", "train", a, a, PairStat(1, 1.0, 1.0))


def test_self_play_counts_for_no_entity_and_games_count_comparisons():
    table = ComparisonTable()
    for ranks in ([1.0, 2.0], [2.0, 1.0]):
        table.add_result(make_match_result([(0, "main", "latest", "player", 3), (1, "main", "latest", "player", 4)],
                                           ranks), [0, 0])          # one bucket against itself: only the side bias
    table.add_result(make_match_result([(0, "main", "latest", "player", 15), (1, "random", "fixed")]), [10, None])
    assert table.entity_games("2p", ["train"]) == {"main@v10": 1, "random": 1}
    ((a, b, stat), _) = sorted(table.pairs("2p", ["train"]), key=lambda r: r[2].n, reverse=True)
    assert a.entities() == b.entities() == ["main@v0"] and stat == PairStat(2, 2.0, 1.0)
    table.add_result(make_match_result([(t, f"p{t}", "fixed") for t in range(3)], layout="3p"), [None] * 3)
    assert table.entity_games("3p", ["train"]) == {"p0": 2, "p1": 2, "p2": 2}      # T - 1 comparisons per match


def test_score_layouts_keep_n_sum_and_sum_of_squares():
    table = ComparisonTable()
    for score in (1.0, 3.0):
        table.add_result(make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[score], layout="coop"),
                         [None, None])
    ((key, stat),) = table.scores("coop", ["train"])
    assert key.entities() == ["a", "b"] and stat == ScoreStat(2, 4.0, 10.0)
    assert table.pairs("coop", ["train"]) == [] and table.entity_games("coop", ["train"]) == {"a": 2, "b": 2}


def test_copy_is_independent_and_merge_adds_everything():
    table = ComparisonTable()
    table.add_result(make_match_result([(0, "a", "fixed"), (1, "b", "fixed")]), [None, None])
    copy = table.copy()
    table.add_result(make_match_result([(0, "a", "fixed"), (1, "b", "fixed")]), [None, None])
    assert copy.pairs("2p", ["train"])[0][2].n == 1 and table.pairs("2p", ["train"])[0][2].n == 2
    version = copy.version
    copy.merge(table)
    assert copy.pairs("2p", ["train"])[0][2] == PairStat(3, 3.0, 3.0) and copy.version > version


def test_concurrent_writers_and_copies_lose_nothing():
    table = ComparisonTable()
    result = make_match_result([(0, "a", "fixed"), (1, "b", "fixed")])

    def writer():
        for _ in range(2000):
            table.add_result(result, [None, None])

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        table.copy()                                    # a reader thread copying while the writers write
    for thread in threads:
        thread.join()
    assert table.pairs("2p", ["train"])[0][2].n == 4000
