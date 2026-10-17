"""Tournament statistics (SP4 T5.3, spec block 3): candidate scores, lineage weights, the paired stratified
bootstrap, confirmed, repeated games."""
from __future__ import annotations

import numpy as np
import pytest

from colosseum.arena.tournament.schedule import gauntlet_blocks, score_blocks
from colosseum.arena.tournament.stats import SELF, candidate_scores, combine, confirmed, select_winner
from game_helpers import CoopGame, EliminationFFA, FakeArena, TurnTakingGame, synthetic_players

PLAYERS = synthetic_players(["main@v10", "main@v20", "other@v5", "greedy", "random", "main@v1", "main@v2"])
NAMES = {(p.agent_id, p.network_id): n for n, p in PLAYERS.items()}
LINEAGE = {n: p.lineage for n, p in PLAYERS.items()}


def tables_for(blocks, results, *, deterministic=False):
    return candidate_scores(blocks, results, names=NAMES, lineage=LINEAGE,
                            group_of={(b.layout, b.candidate): "player" for b in blocks},
                            metric_of={"2p": "pairwise", "4p": "pairwise", "coop2": "score"},
                            deterministic=deterministic)


def test_scores_are_mean_pairwise_scores_with_equal_lineage_weights():
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy", "main@v1", "main@v2"], PLAYERS, blocks=2, seed=0,
                             stage="A")
    strengths = {"main@v10": 0.0, "greedy": -100.0, "main@v1": 100.0, "main@v2": 100.0}
    (table,) = tables_for(blocks, FakeArena(spec, PLAYERS, strengths)(blocks)).values()
    assert table.columns == ["greedy", "main@v1", "main@v2"]
    assert table.cells().tolist() == [[1.0, 0.0, 0.0]]
    assert table.weights.tolist() == [0.5, 0.25, 0.25]     # lineages greedy and main: the snapshots share a weight
    assert table.scores().tolist() == [0.5]
    assert table.games["main@v10"] == {"greedy": 4, "main@v1": 4, "main@v2": 4} and table.blocks_of("greedy") == 2


def test_ffa_pairwise_scores_count_every_opposing_team():
    spec = EliminationFFA(4).spec
    blocks = gauntlet_blocks(spec, "4p", ["main@v10"], ["greedy", "random"], PLAYERS, blocks=1, seed=0, stage="A")
    arena = FakeArena(spec, PLAYERS, {"main@v10": 50.0, "greedy": 0.0, "random": -50.0})
    (table,) = tables_for(blocks, arena(blocks)).values()
    assert table.cells().tolist() == [[1.0, 1.0]] and table.scores().tolist() == [1.0]
    # four multisets of three field members, four rotations each: matches, not pairwise comparisons
    assert table.matches["main@v10"] == 16 and sum(table.games["main@v10"].values()) == 48


def test_paired_bootstrap_confirms_only_a_real_gap():
    spec = TurnTakingGame().spec
    strengths = {"main@v10": 1.0, "main@v20": 0.0, "other@v5": 0.0, "greedy": 0.0, "random": -0.5}
    blocks = gauntlet_blocks(spec, "2p", ["main@v10", "main@v20", "other@v5"], ["greedy", "random"], PLAYERS,
                             blocks=150, seed=0, stage="B")
    tables = list(tables_for(blocks, FakeArena(spec, PLAYERS, strengths, seed=1)(blocks)).values())
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=1000, rng=np.random.default_rng(2)), tables)
    assert sel.winner == "main@v10" and sel.confirmed and sel.diff_ci95[0] > 0
    tables = list(tables_for(blocks, FakeArena(spec, PLAYERS, {**strengths, "main@v10": 0.0}, seed=3)(blocks)).values())
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=1000, rng=np.random.default_rng(4)), tables)
    assert not sel.confirmed and sel.runner_up in sel.indistinguishable


def test_the_bootstrap_interval_covers_the_true_score():
    """Coverage of the per-candidate 95 % interval over 200 synthetic gauntlets of 60 blocks per composition; the
    true score is the BT expectation (Gumbel performance: P(win) = sigmoid(difference)). Measured coverage of this
    percentile bootstrap: 0.936 over 1500 runs; with n = 200 a correct implementation gives >= 176 hits except
    with probability < 1e-3 (binomial, p = 0.936, z = -3.1). The seeds are fixed, so the test is deterministic."""
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy", "random"], PLAYERS, blocks=60, seed=0, stage="A")
    strengths = {"main@v10": 0.5, "greedy": 0.0, "random": -1.0}
    truth = 0.5 / (1 + np.exp(-0.5)) + 0.5 / (1 + np.exp(-1.5))
    arena = FakeArena(spec, PLAYERS, strengths, seed=5)
    rng = np.random.default_rng(6)
    hits = 0
    for _ in range(200):
        (table,) = tables_for(blocks, arena(blocks)).values()
        lo, hi = np.quantile(table.bootstrap_scores(400, rng)[:, 0], [0.025, 0.975])
        hits += lo <= truth <= hi
    assert hits >= 176


def test_repeated_deterministic_games_are_flagged_and_never_confirm():
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10", "main@v20"], ["greedy"], PLAYERS, blocks=5, seed=0, stage="B")
    results = FakeArena(spec, PLAYERS, {"main@v10": 100.0, "main@v20": -100.0, "greedy": 0.0})(blocks)
    tables = list(tables_for(blocks, results, deterministic=True).values())
    assert tables[0].repeated == {("main@v10", "greedy"), ("main@v20", "greedy")}
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=200, rng=np.random.default_rng(1)), tables)
    assert sel.winner == "main@v10" and not sel.confirmed and sel.repeated_columns == [("2p", "greedy")]
    assert not list(tables_for(blocks, results, deterministic=False).values())[0].repeated


def test_score_layouts_average_team_scores_over_the_homogeneous_team_and_the_field():
    spec = CoopGame(size=2).spec
    blocks = score_blocks(spec, "coop2", ["main@v10"], ["greedy"], PLAYERS, blocks=3, seed=0, stage="A")
    arena = FakeArena(spec, PLAYERS, {"main@v10": 4.0, "greedy": 0.0}, noise=0.0)
    (table,) = tables_for(blocks, arena(blocks)).values()
    assert table.columns == [SELF, "greedy"] and table.cells().tolist() == [[4.0, 2.0]]
    assert table.scores().tolist() == [3.0]


def test_tables_need_the_same_blocks_for_every_candidate():
    spec = TurnTakingGame().spec
    a = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=2, seed=0, stage="A")
    b = gauntlet_blocks(spec, "2p", ["main@v20"], ["greedy"], PLAYERS, blocks=3, seed=0, stage="A")
    results = FakeArena(spec, PLAYERS, {"main@v10": 0, "main@v20": 0, "greedy": 0})(a + b)
    with pytest.raises(ValueError, match="same blocks"):
        tables_for(a + b, results)
    c = gauntlet_blocks(spec, "2p", ["main@v20"], ["random"], PLAYERS, blocks=2, seed=0, stage="A")
    results = FakeArena(spec, PLAYERS, {"main@v10": 0, "main@v20": 0, "greedy": 0, "random": 0})(a + c)
    with pytest.raises(ValueError, match="same blocks"):
        tables_for(a + c, results)


def test_a_candidate_in_the_field_is_scored_without_itself():
    """T5.2 leaves a candidate that is also a field member out of its own compositions: its row averages the
    other columns (lineage weights renormalized), the other candidates keep every column."""
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10", "greedy"], ["greedy", "random"], PLAYERS, blocks=3, seed=0,
                             stage="A")
    arena = FakeArena(spec, PLAYERS, {"main@v10": 100.0, "greedy": 0.0, "random": -100.0})
    (table,) = tables_for(blocks, arena(blocks)).values()
    assert table.candidates == ["main@v10", "greedy"] and table.columns == ["greedy", "random"]
    assert table.has_cell("main@v10", "greedy") and not table.has_cell("greedy", "greedy")
    assert table.scores().tolist() == [1.0, 1.0]           # greedy: only random, which it always beats
    assert table.candidate_weights.tolist() == [[0.5, 0.5], [0.0, 1.0]]
    assert table.self_excluded == ["greedy"] and table.blocks_of("greedy", "greedy") == 0
    reps = table.bootstrap_scores(50, np.random.default_rng(0))
    assert reps.shape == (50, 2) and np.allclose(reps[:, 1], 1.0)


def test_confirmed_needs_the_interval_above_zero_and_no_conflict():
    rng = np.random.default_rng(0)
    gap = rng.normal(0.2, 0.05, size=2000)
    assert confirmed(gap) and not confirmed(gap, conflict=True) and not confirmed(gap - 0.2)


def test_a_lone_finalist_is_not_confirmed():
    spec = TurnTakingGame().spec
    blocks = gauntlet_blocks(spec, "2p", ["main@v10"], ["greedy"], PLAYERS, blocks=5, seed=0, stage="B")
    tables = list(tables_for(blocks, FakeArena(spec, PLAYERS, {"main@v10": 3.0, "greedy": 0.0})(blocks)).values())
    sel = select_winner(combine(tables, {"2p": 1.0}, samples=100, rng=np.random.default_rng(0)), tables)
    assert sel.winner == "main@v10" and sel.runner_up is None and sel.confirmed is False and sel.diff_ci95 is None


def test_confirmed_is_rarely_claimed_between_equal_finalists():
    """``confirmed`` is honest: with two equally strong finalists (30 blocks per composition, two field members)
    it holds in about 6 % of tournaments (measured: 0.061 and 0.063 over 1000 runs each; nominal 5 %, the
    winner is the larger point estimate, so both tails count). Over 200 runs a correct implementation gives at
    most 24 confirmations except with probability < 1e-3 (binomial, p = 0.063, z = 3.5). Fixed seeds."""
    spec = TurnTakingGame().spec
    players = synthetic_players(["main@v1", "main@v2", "greedy", "random"])
    names = {(p.agent_id, p.network_id): n for n, p in players.items()}
    lineage = {n: p.lineage for n, p in players.items()}
    blocks = gauntlet_blocks(spec, "2p", ["main@v1", "main@v2"], ["greedy", "random"], players, blocks=30, seed=0,
                             stage="B")
    arena = FakeArena(spec, players, {"main@v1": 0.0, "main@v2": 0.0, "greedy": 0.0, "random": -0.5}, seed=11)
    rng = np.random.default_rng(12)
    claims = 0
    for _ in range(200):
        tables = list(candidate_scores(blocks, arena(blocks), names=names, lineage=lineage,
                                       group_of={(b.layout, b.candidate): "player" for b in blocks},
                                       metric_of={"2p": "pairwise"}).values())
        claims += select_winner(combine(tables, {"2p": 1.0}, samples=400, rng=rng), tables).confirmed
    assert claims <= 24
