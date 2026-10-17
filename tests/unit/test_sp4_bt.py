"""Bradley-Terry ratings (SP4 T2.1, spec block 2, criterion 3.2 and section 6 "Bradley–Terry"): coverage of the
true anchored differences on synthetic data (1v1 with a side and draws, teams, FFA, asymmetric roles with and
without a bridge, a disconnected graph), the side bias, self-play, anchors, smoothness, finite estimates, the G2
test (its level under the null and its power on a cycle), score layouts, the iterative path (perturb-and-solve
uncertainty), warm starts, Laplace and bootstrap uncertainty (paired keys, independent strata, coverage of the
table), and fits from the journal before and after compaction. Every test reads the module's prior and smoothness
constants (T8.2 may change them)."""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from colosseum.arena.bt import (
    ELO_PER_LOGIT,
    MEAN_OF_FIXED,
    Z_95,
    BootstrapData,
    BTOptions,
    bootstrap,
    chi2_sf,
    fit,
    laplace,
    score_summary,
    transitivity_test,
)
from colosseum.arena.comparisons import ComparisonTable, PairStat, match_pairs
from colosseum.arena.matchlog import MatchLog, RunInfo
from game_helpers import bt_table, make_match_result, synthetic_bt_matches, synthetic_journal_results

TRAIN = ("train",)


# ---------------------------------------------------------------------------
# coverage scenarios: each returns (true strengths in logits, results, fit keyword arguments)
# ---------------------------------------------------------------------------


def _one_v_one(rng):
    strengths = {"random": -2.0, "greedy": 0.5, "frozen": 1.0}
    strengths.update({f"main@v{v}": -1.5 + 0.25 * k for k, v in enumerate(range(0, 1300, 100))})
    results = synthetic_bt_matches(strengths, n_matches=140 * len(strengths), rng=rng, bias=[0.4, -0.4], draw=0.2)
    return strengths, results, dict(fixed_agents=("random", "greedy", "frozen"), random_bot="random",
                                    agent_order=("main", "random", "greedy", "frozen"))


def _teams(rng):
    strengths = {f"a{i}": (i - 7) * 0.3 for i in range(16)}
    strengths["random"] = -2.5
    return strengths, synthetic_bt_matches(strengths, n_matches=1700, rng=rng, team_size=2, bias=[0.2, -0.2]), dict(
        fixed_agents=("random",), random_bot="random")


def _ffa(rng):
    strengths = {f"a{i}": (i - 6) * 0.3 for i in range(12)}
    strengths["random"] = -2.0
    results = synthetic_bt_matches(strengths, n_matches=900, rng=rng, num_teams=4, bias=[0.3, 0.1, -0.1, -0.3],
                                   layout="4p")
    return strengths, results, dict(fixed_agents=("random",), random_bot="random")


def _asymmetric(bridge: bool):
    def scenario(rng):
        hunters = {f"h{i}": (i - 2) * 0.5 for i in range(5)}
        prey = {f"p{i}": (i - 2) * 0.5 for i in range(5)}
        strengths = {**hunters, **prey, **({"random": -1.5} if bridge else {})}
        h_pool = sorted(hunters) + (["random"] if bridge else [])
        p_pool = sorted(prey) + (["random"] if bridge else [])

        def choose(rng):
            while True:
                h, p = h_pool[rng.integers(len(h_pool))], p_pool[rng.integers(len(p_pool))]
                if h != p:
                    return [[h], [p]]

        results = synthetic_bt_matches(strengths, n_matches=1800, rng=rng, bias=[0.7, -0.7],
                                       roles=["hunter", "prey"], layout="1v1", choose=choose)
        kwargs = dict(fixed_agents=("random",), random_bot="random") if bridge else {}
        return strengths, results, kwargs
    return scenario


def _disconnected(rng):
    strengths = {f"a{i}": i * 0.4 for i in range(4)} | {f"b{i}": 3.0 + i * 0.4 for i in range(4)}

    def choose(rng):
        side = "a" if rng.random() < 0.5 else "b"
        i, j = rng.choice(4, 2, replace=False)
        return [[f"{side}{i}"], [f"{side}{j}"]]

    return strengths, synthetic_bt_matches(strengths, n_matches=1200, rng=rng, choose=choose), {}


SCENARIOS = {"1v1_side_draws": _one_v_one, "2v2_teams": _teams, "ffa_4": _ffa,
             "asymmetric_no_bridge": _asymmetric(False), "asymmetric_bridge": _asymmetric(True),
             "disconnected": _disconnected}
REPLICATIONS = 20
MIN_COVERAGE = 0.85


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_true_anchored_differences_fall_in_the_95_percent_intervals(name):
    """Criterion 3.2. Over 20 replications (seeds 0..19) the true difference of every non-anchor entity to its
    cell's anchor (same scale) must fall into its Laplace 95 % interval at least 85 % of the time (nominal 0.95,
    tolerance 0.10). Every entity has at least 200 comparisons; prior and smoothness are the module's. Calibration
    (T2.1, 15 independent runs of 20 replications per scenario, seeds 1000k..1000k+19): mean coverage 0.945-0.990,
    run-to-run sd at most 0.030, so 0.85 lies at least 3.4 sd below every scenario's mean: a correct implementation
    fails with probability below 1e-3 (normal approximation, per scenario). That calibration used the provisional
    prior / smoothness 4.0 / 0.5; with the T8.2 values 8.0 / 1.0 one run of 20 replications (seeds 0..19) gave nearly
    the same per-scenario coverage: 0.977 / 0.966 / 0.996 / 0.950 / 0.950 / 0.967 (4.0 / 0.5: 0.977 / 0.963 / 0.996 /
    0.950 / 0.950 / 0.967)."""
    hits = total = 0
    for seed in range(REPLICATIONS):
        strengths, results, kwargs = SCENARIOS[name](np.random.default_rng(seed))
        fitted = fit(bt_table(results), results[0].layout, sources=TRAIN, **kwargs)
        assert fitted.converged
        assert min(fitted.games.values()) >= 200
        for group in fitted.groups:
            anchor = np.mean([strengths[a] for a in group.anchor_members])
            for entity in group.entities:
                if group.anchor_members == [entity]:
                    continue
                true = (strengths[entity] - anchor) * ELO_PER_LOGIT
                half = Z_95 * fitted.se[entity]
                hits += fitted.rating[entity] - half <= true <= fitted.rating[entity] + half
                total += 1
    assert hits / total >= MIN_COVERAGE, (hits, total)


def test_groups_without_a_bridge_are_not_compared_and_a_bridge_joins_them():
    _, results, _ = _asymmetric(False)(np.random.default_rng(0))
    alone = fit(bt_table(results), "1v1", sources=TRAIN)
    assert [(g.group, g.component) for g in alone.groups] == [("hunter", 0), ("prey", 0)]
    assert {g.group: g.anchor for g in alone.groups} == {"hunter": "h0", "prey": "p0"}   # each group its own zero
    assert alone.rating["h0"] == alone.rating["p0"] == 0.0
    _, results, kwargs = _asymmetric(True)(np.random.default_rng(0))
    bridged = fit(bt_table(results), "1v1", sources=TRAIN, **kwargs)
    ((group,),) = [bridged.groups]
    assert group.group == "hunter+prey" and group.anchor == "random" and len(group.entities) == 11


def test_a_disconnected_graph_gets_an_anchor_per_component():
    _, results, _ = _disconnected(np.random.default_rng(0))
    fitted = fit(bt_table(results), "2p", sources=TRAIN)
    assert [(g.component, g.anchor, g.entities) for g in fitted.groups] == [
        (0, "a0", ["a0", "a1", "a2", "a3"]), (1, "b0", ["b0", "b1", "b2", "b3"])]
    assert fitted.rating["a0"] == fitted.rating["b0"] == 0.0


def test_the_side_bias_is_recovered_and_sums_to_zero():
    """1v1 without draws, true bias +-0.4 logits; the estimate is within 4 standard errors (a correct
    implementation fails with probability 6e-5)."""
    strengths = {f"a{i}": (i - 4) * 0.3 for i in range(9)}
    results = synthetic_bt_matches(strengths, n_matches=6000, rng=np.random.default_rng(3), bias=[0.4, -0.4])
    fitted = fit(bt_table(results), "2p", sources=TRAIN)
    assert fitted.bias[0] == pytest.approx(-fitted.bias[1])
    assert abs(fitted.bias[0] - 0.4 * ELO_PER_LOGIT) < 4 * fitted.bias_se[0]


def test_self_play_informs_only_the_bias_and_counts_for_no_games():
    """3000 self-play matches (first side wins 70 %) inform only the bias; the estimate lies within 4 of its
    standard errors of the truth (normal approximation: a correct implementation fails with probability 6e-5)."""
    rng = np.random.default_rng(5)
    results = []
    for k in range(3000):                       # latest of one bucket against itself: the first side wins 70 %
        first = rng.random() < 0.7
        results.append(make_match_result([(0, "main", "latest", "player", 3), (1, "main", "latest", "player", 4)],
                                         [1.0, 2.0] if first else [2.0, 1.0], uid=f"s{k}"))
    for k in range(400):
        results.append(make_match_result([(0, "main", "latest", "player", 2), (1, "random", "fixed")],
                                         [1.0, 2.0] if rng.random() < 0.6 else [2.0, 1.0], uid=f"r{k}"))
    fitted = fit(bt_table(results, interval=10), "2p", sources=TRAIN, random_bot="random", fixed_agents=("random",))
    assert fitted.games == {"main@v0": 400, "random": 400}
    true_bias = 0.5 * math.log(0.7 / 0.3) * ELO_PER_LOGIT        # b0 - b1 = logit(0.7)
    assert abs(fitted.bias[0] - true_bias) < 4 * fitted.bias_se[0]


def _duels(pairs: dict[tuple[str, str], tuple[int, int]]) -> ComparisonTable:
    """A table from ``{(a, b): (wins of a, wins of b)}``, every match on both sides equally."""
    table = ComparisonTable()
    k = 0
    for (a, b), (wins_a, wins_b) in pairs.items():
        for winner, count in ((a, wins_a), (b, wins_b)):
            for i in range(count):
                first, second = (a, b) if i % 2 == 0 else (b, a)
                table.add_result(make_match_result([(0, *_seat(first)), (1, *_seat(second))],
                                                   [1.0, 2.0] if winner == first else [2.0, 1.0], uid=f"d{k}"),
                                 [_version(first), _version(second)])
                k += 1
    return table


def _seat(entity: str) -> tuple[str, str]:
    if "@v" in entity:
        agent, version = entity.rsplit("@v", 1)
        return agent, f"ckpt_v{version}"
    return entity, "fixed"


def _version(entity: str) -> int | None:
    return int(entity.rsplit("@v", 1)[1]) if "@v" in entity else None


def test_anchor_rules():
    table = _duels({("main@v0", "random"): (30, 30), ("main@v100", "greedy"): (40, 20), ("random", "greedy"): (20, 40),
                    ("other@v0", "main@v100"): (10, 50)})
    order = ("main", "other", "random", "greedy")
    by_rule = {}
    for options, kwargs in (
            (BTOptions(anchor="greedy"), {}),
            (BTOptions(anchor="main"), {}),
            (BTOptions(), dict(random_bot="random")),
            (BTOptions(), dict(fixed_agents=("random", "greedy"))),
            (BTOptions(), dict(agent_order=order)),
            (BTOptions(), {}),
            (BTOptions(anchor="nobody"), dict(random_bot="random")),
    ):
        fitted = fit(table, "2p", sources=TRAIN, options=options, **kwargs)
        (group,) = fitted.groups
        by_rule[(options.anchor, tuple(sorted(kwargs)))] = (group.anchor, group.rule, group.note)
        members = group.anchor_members
        assert np.mean([fitted.rating[m] for m in members]) == pytest.approx(0.0, abs=1e-9)
    assert by_rule[("greedy", ())] == ("greedy", "config", None)
    assert by_rule[("main", ())] == ("main@v0", "config", None)                 # a trainable agent: its v0
    assert by_rule[(None, ("random_bot",))] == ("random", "random_bot", None)
    assert by_rule[(None, ("fixed_agents",))] == (MEAN_OF_FIXED, "mean_of_fixed", None)
    assert by_rule[(None, ("agent_order",))] == ("main@v0", "first_trainable", None)
    assert by_rule[(None, ())] == ("greedy", "first", None)
    assert by_rule[("nobody", ("random_bot",))] == ("random", "random_bot", "ratings.anchor 'nobody' did not play here")


def test_a_random_bot_that_did_not_play_is_replaced_with_a_note():
    table = _duels({("main@v0", "greedy"): (30, 30), ("main@v100", "greedy"): (40, 20)})
    (group,) = fit(table, "2p", sources=TRAIN, random_bot="random", fixed_agents=("random", "greedy")).groups
    assert (group.anchor, group.rule, group.note) == (MEAN_OF_FIXED, "mean_of_fixed",
                                                      "RandomBot agent 'random' did not play here")
    assert group.anchor_members == ["greedy"]


def test_smoothness_pulls_a_rarely_played_version_towards_its_neighbours():
    pairs = {(f"main@v{v}", "random"): (50 + 2 * k, 50 - 2 * k) for k, v in enumerate((0, 10, 20, 40, 50))}
    pairs[("main@v30", "random")] = (0, 3)                      # three games, all lost
    table = _duels(pairs)
    smooth = fit(table, "2p", sources=TRAIN, random_bot="random")
    rough = fit(table, "2p", sources=TRAIN, random_bot="random", options=BTOptions(smoothness=0.0))
    neighbours = (smooth.rating["main@v20"] + smooth.rating["main@v40"]) / 2
    assert abs(smooth.rating["main@v30"] - neighbours) < abs(rough.rating["main@v30"] - neighbours)
    assert smooth.rating["main@v30"] > rough.rating["main@v30"]


def test_an_entity_that_always_loses_or_rarely_plays_gets_a_finite_rating():
    table = _duels({("x", "loser"): (40, 0), ("y", "loser"): (40, 0), ("x", "y"): (20, 20), ("x", "rare"): (1, 0)})
    fitted = fit(table, "2p", sources=TRAIN, fixed_agents=("x", "y", "loser", "rare"))
    for entity in ("loser", "rare"):
        assert math.isfinite(fitted.rating[entity]) and math.isfinite(fitted.se[entity]) and fitted.se[entity] > 0
    assert fitted.rating["loser"] < fitted.rating["x"] - 300


def test_the_transitivity_test_is_quiet_on_transitive_data_with_rare_pairs_and_flags_a_cycle():
    """Transitive: 6 entities with about 100 matches per pair and side, plus 10 entities with a few matches each
    (pairs below min_games are left out); under the null the p-value is approximately uniform (asymptotic chi^2 of
    the likelihood-ratio statistic), so the assertion p > 1e-4 fails with probability 1e-4 at the nominal level;
    the measured level is about 1.7 times nominal (test below), and even a 5-fold excess in the tail keeps the
    failure probability at 5e-4 < 1e-3 (seed 0 was checked). A rock-paper-scissors cycle (80 % each way, 300
    matches per pair) is flagged."""
    rng = np.random.default_rng(0)
    strengths = {f"a{i}": i * 0.3 for i in range(6)}

    def frequent(rng):
        i, j = rng.choice(6, 2, replace=False)
        return [[f"a{i}"], [f"a{j}"]]

    results = synthetic_bt_matches(strengths, n_matches=3000, rng=rng, choose=frequent, bias=[0.3, -0.3])
    rare = {f"r{i}": float(rng.normal()) for i in range(10)}
    results += synthetic_bt_matches({**strengths, **rare}, n_matches=100, rng=rng, uid_prefix="rare",
                                    choose=lambda rng: [[f"r{rng.integers(10)}"], [f"a{rng.integers(6)}"]])
    table = bt_table(results)
    transitive = transitivity_test(table, fit(table, "2p", sources=TRAIN), sources=TRAIN)
    assert transitive.pairs_tested == 30 and transitive.dof > 0
    assert transitive.p_value > 1e-4 and transitive.flagged == (transitive.p_value < 0.01)
    cycle = _duels({("A", "B"): (240, 60), ("B", "C"): (240, 60), ("C", "A"): (240, 60)})
    result = transitivity_test(cycle, fit(cycle, "2p", sources=TRAIN), sources=TRAIN)
    assert result.flagged and result.p_value < 1e-10 and len(result.worst_pairs) == 5
    few = transitivity_test(cycle, fit(cycle, "2p", sources=TRAIN), sources=TRAIN, min_games=10_000)
    assert (few.p_value, few.flagged, few.pairs_tested) == (None, False, 0)


def test_chi2_survival_function():
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, rel=1e-6)
    assert chi2_sf(18.307038053275146, 10) == pytest.approx(0.05, rel=1e-6)
    assert chi2_sf(124.34211340400407, 100) == pytest.approx(0.05, rel=1e-6)
    assert chi2_sf(0.0, 3) == 1.0 and chi2_sf(1e4, 3) < 1e-300


def test_json_forms_options_and_bootstrap_inputs(tmp_path):
    """Contract amendments A10-A12: ``default_options``, ``BTOptions.agent_order``, ``TransitivityResult.to_dict``,
    ``ScoreSummary.to_dict`` / ``by_entity``, ``BootstrapData.from_table`` / ``from_blocks`` and
    ``bootstrap_data`` over a journal."""
    import json

    from colosseum.arena.bt import bootstrap_data, default_options

    assert default_options("rnd") == BTOptions(anchor="rnd")
    duels = _duels({("main@v0", "main@v10"): (30, 30), ("main@v10", "main@v20"): (30, 30)})
    (group,) = fit(duels, "2p", sources=TRAIN, options=BTOptions(agent_order=("main",))).groups
    assert (group.anchor, group.rule) == ("main@v0", "first_trainable")
    cycle = _duels({("A", "B"): (240, 60), ("B", "C"): (240, 60), ("C", "A"): (240, 60)})
    d = transitivity_test(cycle, fit(cycle, "2p", sources=TRAIN), sources=TRAIN).to_dict()
    assert d["flag"] and d["worst_pairs"][0]["games"] == 150 and json.dumps(d)
    table = ComparisonTable()
    for k, score in enumerate((1.0, 3.0)):
        table.add_result(make_match_result([(0, "a", "fixed")], scores=[score], layout="coop", uid=f"c{k}"), [None])
    summary = score_summary(table, "coop", sources=TRAIN)
    n, mean, se = summary.by_entity["a"]
    assert (n, mean) == (2, 2.0) and se == pytest.approx(math.sqrt(2.0) / math.sqrt(2.0))
    assert summary.to_dict()["entities"]["a"]["n"] == 2 and json.dumps(summary.to_dict())
    assert len(BootstrapData.from_table(cycle, "2p", sources=TRAIN).counts) == len(cycle.pairs("2p", TRAIN))
    block = [make_match_result([(0, "main", "ckpt_v10"), (1, "rnd", "fixed")], uid=f"t/A/2p/main@v10/rnd/{i}/r0",
                               block=f"t/A/2p/main@v10/rnd/{i}", source="tournament") for i in range(3)]
    by_block = {r.block: [r] for r in block}
    data = BootstrapData.from_blocks(by_block, strata={b: "A/rnd" for b in by_block})
    assert set(data.blocks) == {"A/rnd"} and sorted(data.blocks["A/rnd"]) == ["0", "1", "2"]
    log = MatchLog(tmp_path / "m.db", RunInfo("R", "run", None, "ttt", "h", 10))
    log.append(synthetic_journal_results(30))
    units = bootstrap_data(log, "2p", sources=TRAIN)
    assert len(units.matches) == 30 and units.counts == [] and units.blocks == {}
    # to_table: the table the data decomposes, equal to the journal's own table, so fit + bootstrap agree
    from colosseum.arena.bt import DATA_SOURCE

    rebuilt, journal = units.to_table("2p"), log.comparison_table(sources=TRAIN)
    assert rebuilt.sources("2p") == [DATA_SOURCE]
    assert ([(a, b, (s.n, s.sw, s.swy)) for a, b, s in rebuilt.pairs("2p", (DATA_SOURCE,))]
            == [(a, b, (s.n, s.sw, s.swy)) for a, b, s in journal.pairs("2p", TRAIN)])
    refit = fit(rebuilt, "2p", sources=(DATA_SOURCE,))
    assert bootstrap(refit, units, samples=2, rng=np.random.default_rng(0)).ci95
    log.close()


def test_score_layouts_summarize_entities_and_compositions():
    table = ComparisonTable()
    for k, (members, score) in enumerate([(("a", "a"), 1.0), (("a", "a"), 3.0), (("a", "b"), 5.0), (("a", "b"), 7.0)]):
        table.add_result(make_match_result([(0, m, "fixed") for m in members], scores=[score], layout="coop",
                                           uid=f"c{k}"), [None, None])
    summary = score_summary(table, "coop", sources=TRAIN)
    assert summary.compositions["a+a"].mean == 2.0 and summary.homogeneous == {"a+a": True, "a+b": False}
    assert (summary.entities["a"].n, summary.entities["a"].mean) == (4, 4.0)
    assert (summary.entities["b"].n, summary.entities["b"].mean) == (2, 6.0)
    half = Z_95 * math.sqrt(2.0 / 2)                           # sd of (5, 7) = sqrt(2), n = 2
    assert summary.compositions["a+b"].ci_low == pytest.approx(6.0 - half)
    assert fit(table, "coop", sources=TRAIN).groups == []      # no pairs: no Bradley-Terry fit


def test_the_iterative_path_agrees_with_the_dense_one():
    """Ratings agree to the solver tolerance. The iterative standard errors are Monte Carlo estimates from
    ``ITERATIVE_SAMPLES`` (400) exact draws of N(0, H^-1): relative sd about 1 / sqrt(2 * 400) = 3.5 %, so the
    band 0.85-1.15 lies 4.3 sd away; over 16 entities a correct implementation fails with probability < 1e-3."""
    strengths, results, kwargs = _one_v_one(np.random.default_rng(1))
    table = bt_table(results)
    dense = fit(table, "2p", sources=TRAIN, **kwargs)
    iterative = fit(table, "2p", sources=TRAIN, options=BTOptions(dense_limit=5), **kwargs)
    assert not dense.approximate and iterative.approximate and iterative.converged
    assert max(abs(dense.rating[e] - iterative.rating[e]) for e in dense.rating) < 1e-6
    assert all(0.85 < iterative.se[e] / dense.se[e] < 1.15 for e in dense.se if dense.se[e] > 0)
    assert len(laplace(iterative, samples=200, rng=np.random.default_rng(0)).lcb) == len(strengths)


def test_a_warm_start_converges_faster_to_the_same_fit():
    _, results, kwargs = _one_v_one(np.random.default_rng(2))
    table = bt_table(results)
    cold = fit(table, "2p", sources=TRAIN, **kwargs)
    warm = fit(table, "2p", sources=TRAIN, warm=cold, **kwargs)
    assert warm.iterations < cold.iterations
    assert max(abs(cold.rating[e] - warm.rating[e]) for e in cold.rating) < 1e-6


def test_an_empty_table_gives_an_empty_fit():
    empty = fit(ComparisonTable(), "2p", sources=TRAIN)
    assert (empty.groups, empty.rating, empty.converged) == ([], {}, True)
    assert laplace(empty, rng=np.random.default_rng(0)).ci95 == {}
    assert transitivity_test(ComparisonTable(), empty, sources=TRAIN).p_value is None


def test_laplace_probabilities_lower_bounds_and_determinism():
    """Probabilities sum as they must; ``lcb`` (the 10 % quantile of a normal lineage contrast centred at the MAP
    contrast) lies 1.28 sd below the centre, and its Monte Carlo sd with 2000 draws is about 0.04 sd, so the
    "below the mean contrast" assertion fails only after a 30-sd deviation (probability far below 1e-3)."""
    _, results, kwargs = _one_v_one(np.random.default_rng(4))
    fitted = fit(bt_table(results), "2p", sources=TRAIN, **kwargs)
    first = laplace(fitted, samples=2000, rng=np.random.default_rng(7), top_k=3, quantile=0.1)
    again = laplace(fitted, samples=2000, rng=np.random.default_rng(7), top_k=3, quantile=0.1)
    assert first.lcb == again.lcb
    assert sum(first.p_best_group.values()) == pytest.approx(1.0)
    mains = [e for e in fitted.rating if e.startswith("main@")]
    assert sum(first.p_best_lineage[e] for e in mains) == pytest.approx(1.0)
    assert sum(first.p_top_k[e] for e in mains) == pytest.approx(3.0)
    assert sum(first.p_top_k_group.values()) == pytest.approx(3.0)
    mean_main = np.mean([fitted.rating[e] for e in mains])
    for entity in mains:                                       # the 10 % quantile lies below the mean contrast
        assert first.lcb[entity] < fitted.rating[entity] - mean_main
        assert first.contrast_samples[entity].shape == (2000,)
    lo, hi = first.ci95["main@v1200"]
    assert (hi - lo) / 2 == pytest.approx(Z_95 * fitted.se["main@v1200"])


def test_bootstrap_by_matches_counts_and_blocks():
    """Bootstrap intervals (60 replicates) against Laplace ones: asymptotically the same width; the Monte Carlo
    relative error of a percentile width from 60 replicates is about 12 %, so the band 0.5-2.0 is more than 5 sd
    wide on the log scale; the MAP lies inside a percentile interval unless the 2.5 % quantile of the replicates
    passes their median. A correct implementation fails with probability below 1e-3."""
    strengths, results, kwargs = _one_v_one(np.random.default_rng(6))
    table = bt_table(results)
    fitted = fit(table, "2p", sources=TRAIN, **kwargs)
    fast = laplace(fitted, samples=1000, rng=np.random.default_rng(0))
    contributions = [match_pairs(r, [None if s.network_id == "fixed" else int(s.network_id[6:]) for s in r.seats])
                     for r in results]
    by_matches = bootstrap(fitted, BootstrapData(matches=contributions), samples=60, rng=np.random.default_rng(1))
    by_counts = bootstrap(fitted, BootstrapData(counts=table.pairs("2p", TRAIN)), samples=60,
                          rng=np.random.default_rng(1))
    blocks = {"field": {str(k): [c for match in contributions[k * 8:(k + 1) * 8] for c in match]
                        for k in range(len(contributions) // 8)}}
    by_blocks = bootstrap(fitted, BootstrapData(blocks=blocks), samples=60, rng=np.random.default_rng(1))
    for uncertainty in (by_matches, by_counts, by_blocks):
        for entity in ("main@v0", "main@v1200", "greedy"):
            lo, hi = uncertainty.ci95[entity]
            flo, fhi = fast.ci95[entity]
            assert 0.5 < (hi - lo) / (fhi - flo) < 2.0          # same order of magnitude as Laplace
            assert lo < fitted.rating[entity] < hi
    stranger = BootstrapData(counts=_duels({("p", "q"): (1, 1)}).pairs("2p", TRAIN))
    with pytest.raises(ValueError, match="does not have"):
        bootstrap(fitted, stranger, samples=2, rng=np.random.default_rng(0))


def test_fits_from_the_journal_match_the_in_memory_table_before_and_after_compaction(tmp_path):
    """Spec section 6 ("сжатие не меняет рейтинги"): the journal's table gives the fit of the in-memory table, and
    so does it after most matches were folded into counts (the team index, hence the bias, is kept)."""
    run = RunInfo("R", "run", None, "ttt", "h", 10)
    log = MatchLog(tmp_path / "matches.db", run, disk_free_fn=lambda _: 10**12)
    results = synthetic_journal_results(3000)
    log.append(results)
    memory = ComparisonTable()
    for result in results:
        memory.add_result(result, log.rating_versions(result))
    kwargs = dict(sources=TRAIN, fixed_agents=("random", "greedy"), random_bot="random", agent_order=("main",))
    reference = fit(memory, "2p", **kwargs)
    before = fit(log.comparison_table(TRAIN), "2p", **kwargs)
    log.compact(window_bytes=log.sizes().detail_bytes // 10)
    after = fit(log.comparison_table(TRAIN), "2p", **kwargs)
    log.close()
    for other in (before, after):
        assert max(abs(other.rating[e] - reference.rating[e]) for e in reference.rating) < 1e-6
        assert max(abs(other.se[e] - reference.se[e]) for e in reference.se) < 1e-6
        assert other.bias[0] == pytest.approx(reference.bias[0], abs=1e-6)
    assert PairStat(1, 1.0, 1.0).n == 1


# ---------------------------------------------------------------------------
# review fixes (T2.1 fix round 1)
# ---------------------------------------------------------------------------


def _ring_and_rare(seed: int) -> ComparisonTable:
    """A transitive truth shaped like a training run: 30 entities, a ring of frequent pairs (about 25 matches per
    pair and side: tested) and many rare random pairs (about 7 per pair and side: below ``min_games``) that carry
    most of the matches."""
    rng = np.random.default_rng(seed)
    strengths = {f"a{i:02d}": float(rng.normal()) * 0.8 for i in range(30)}
    names = sorted(strengths)

    def ring(rng):
        i = int(rng.integers(30))
        j = (i + 1) % 30
        return [[names[i]], [names[j]]] if rng.random() < 0.5 else [[names[j]], [names[i]]]

    def rare(rng):
        i, j = rng.choice(30, 2, replace=False)
        return [[names[i]], [names[j]]]

    results = synthetic_bt_matches(strengths, n_matches=1500, rng=rng, choose=ring, uid_prefix="ring")
    results += synthetic_bt_matches(strengths, n_matches=6000, rng=rng, choose=rare, uid_prefix="rare")
    return bt_table(results)


def test_the_transitivity_test_holds_its_level_when_rare_pairs_dominate():
    """Under a transitive truth (30 seeds) the G2 test flags at most 5 runs and the mean p-value stays above 0.25.
    Calibration (seeds 1000-1599): flag rate 10 / 600 = 0.017 at nominal 0.01 (small-sample chi^2), mean p 0.456.
    Even at a flag rate of 0.03, P(more than 5 of 30) = 2.3e-4; the mean of 30 near-uniform p-values (sd 0.29 / sqrt
    30 = 0.053) falls below 0.25 with probability about 5e-5: a correct implementation fails with probability
    below 1e-3. A test that scores the full-data MAP against the tested pairs' degrees of freedom flags about 57 %
    of these runs."""
    p_values, flags = [], 0
    for seed in range(30):
        table = _ring_and_rare(seed)
        result = transitivity_test(table, fit(table, "2p", sources=TRAIN), sources=TRAIN)
        # tested: the ring's rows with >= 20 matches (about 60); identified: 29 entity contrasts + 1 bias
        assert result.pairs_tested >= 50 and result.dof == result.pairs_tested - 30
        p_values.append(result.p_value)
        flags += result.flagged
    assert flags <= 5 and np.mean(p_values) > 0.25, (flags, np.mean(p_values))


def _chain_table(versions: int = 40, *, seed: int = 0, layout: str = "2p") -> ComparisonTable:
    """Every version plays one of its 3 predecessors (latest against recent snapshots); v0 plays ``random``."""
    rng = np.random.default_rng(seed)
    names = [f"main@v{v}" for v in range(versions)]
    strengths = {name: -2.0 + 0.05 * v for v, name in enumerate(names)} | {"random": -2.0}

    def choose(rng):
        i = int(rng.integers(versions))
        other = "random" if i == 0 else names[max(0, i - 1 - int(rng.integers(3)))]
        teams = [[names[i]], [other]]
        return teams if rng.random() < 0.5 else teams[::-1]

    return bt_table(synthetic_bt_matches(strengths, n_matches=versions * 200, rng=rng, choose=choose, layout=layout))


def test_the_iterative_path_keeps_the_covariances_of_a_chain_graph():
    """A chain graph (strong negative correlations between neighbouring versions and the smoothness chain): the
    iterative path's perturb-and-solve standard errors match the dense ones. Each is a Monte Carlo estimate from
    400 exact draws of N(0, H^-1) (relative sd 3.5 %), so 0.85-1.15 lies 4.3 sd away: over 41 entities a correct
    implementation fails with probability below 1e-3 (a diagonal approximation gives a median ratio of about 0.5).
    ``lcb`` (10 % quantile of the lineage contrast) from 400 iterative and 1000 dense draws differs by Monte Carlo
    error of about 0.1 contrast sd; the band 0.5 sd is 5 of those sd (probability < 1e-5 per entity)."""
    table = _chain_table()
    kwargs = dict(sources=TRAIN, random_bot="random")
    dense = fit(table, "2p", **kwargs)
    iterative = fit(table, "2p", options=BTOptions(dense_limit=5), **kwargs)
    assert not dense.approximate and iterative.approximate and iterative.converged
    ratios = [iterative.se[e] / dense.se[e] for e in dense.se if dense.se[e] > 0]
    assert len(ratios) == 40 and 0.85 < min(ratios) and max(ratios) < 1.15, (min(ratios), max(ratios))
    assert 0.85 < iterative.bias_se[0] / dense.bias_se[0] < 1.15
    fast = laplace(dense, samples=1000, rng=np.random.default_rng(0))
    slow = laplace(iterative, samples=1000, rng=np.random.default_rng(0))
    for entity, samples in fast.contrast_samples.items():
        assert slow.contrast_samples[entity].shape == (400,)
        if entity == "random":                    # a one-entity lineage: contrast 0
            continue
        assert abs(slow.lcb[entity] - fast.lcb[entity]) < 0.5 * float(np.std(samples)), entity


def test_a_failed_line_search_is_not_converged_and_bootstrap_counts_unconverged_replicates(monkeypatch, caplog):
    import colosseum.arena.bt as bt

    table = _duels({("x", "y"): (30, 10), ("y", "z"): (25, 15)})
    calls = []
    real = bt._log_post

    def stuck(d, params, sw, swy, options):        # every step looks worse than the start
        calls.append(1)
        return real(d, params, sw, swy, options) if len(calls) == 1 else -math.inf

    monkeypatch.setattr(bt, "_log_post", stuck)
    stalled = fit(table, "2p", sources=TRAIN)
    assert not stalled.converged and stalled.iterations == 1 and not np.any(stalled.params)
    monkeypatch.undo()
    capped = fit(table, "2p", sources=TRAIN, options=BTOptions(max_iter=1))
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.bt"):
        bootstrap(capped, BootstrapData.from_table(table, "2p", sources=TRAIN), samples=5,
                  rng=np.random.default_rng(0))
    assert "5 of 5 replicates did not converge" in caplog.text


def test_bootstrap_data_must_cover_the_whole_fitted_table():
    _, results, kwargs = _one_v_one(np.random.default_rng(6))
    fitted = fit(bt_table(results), "2p", sources=TRAIN, **kwargs)
    half = bt_table(results[: len(results) // 2])
    with pytest.raises(ValueError, match="does not decompose the fitted table"):
        bootstrap(fitted, BootstrapData.from_table(half, "2p", sources=TRAIN), samples=2, rng=np.random.default_rng(0))


def _paired_results(stratum_of: dict[str, str], keys: int = 30) -> tuple[list, dict]:
    """Per key k: ``main@v0`` and ``main@v10`` each play ``random`` with the same outcome (drawn per k); blocks by
    ``stratum_of[candidate]``. Identical data for the two versions: every replicate that resamples them together
    gives them identical ratings (zero lineage contrast)."""
    rng = np.random.default_rng(3)
    results, blocks = [], {}
    for k in range(keys):
        win = rng.random() < 0.6
        for candidate in ("main@v0", "main@v10"):
            agent, version = candidate.split("@v")
            for side in (0, 1):                  # both sides, so the bias is identified
                seats = [(0, agent, f"ckpt_v{version}"), (1, "random", "fixed")]
                if side:
                    seats = [(0, "random", "fixed"), (1, agent, f"ckpt_v{version}")]
                ranks = [1.0, 2.0] if win != bool(side) else [2.0, 1.0]
                result = make_match_result(seats, ranks, uid=f"{candidate}/{k}/{side}")
                results.append(result)
                versions = [None if s.network_id == "fixed" else int(version) for s in result.seats]
                blocks.setdefault(stratum_of[candidate], {}).setdefault(str(k), []).extend(
                    match_pairs(result, versions))
    return results, blocks


def test_a_block_key_is_drawn_once_for_every_candidate_and_strata_are_independent():
    results, shared = _paired_results({"main@v0": "field", "main@v10": "field"})
    table = ComparisonTable()
    for result in results:
        table.add_result(result, [None if s.network_id == "fixed" else int(s.network_id[6:]) for s in result.seats])
    fitted = fit(table, "2p", sources=TRAIN, random_bot="random")
    assert abs(fitted.rating["main@v0"] - fitted.rating["main@v10"]) < 1e-6
    paired = bootstrap(fitted, BootstrapData(blocks=shared), samples=40, rng=np.random.default_rng(0))
    assert float(np.max(np.abs(paired.contrast_samples["main@v0"]))) < 1e-6      # same draw for both candidates
    _, split = _paired_results({"main@v0": "A", "main@v10": "B"})
    independent = bootstrap(fitted, BootstrapData(blocks=split), samples=40, rng=np.random.default_rng(0))
    assert float(np.std(independent.contrast_samples["main@v0"])) > 1.0          # Elo; strata drawn separately


def test_lineage_probabilities_are_per_cell_when_a_lineage_spans_two_components():
    table = _duels({("main@v0", "a"): (30, 30), ("main@v10", "a"): (35, 25),
                    ("main@v20", "b"): (30, 30), ("main@v30", "b"): (25, 35)})
    fitted = fit(table, "2p", sources=TRAIN, agent_order=("main",))
    assert [(g.component, g.entities) for g in fitted.groups] == [
        (0, ["a", "main@v0", "main@v10"]), (1, ["b", "main@v20", "main@v30"])]
    uncertainty = laplace(fitted, samples=500, rng=np.random.default_rng(0))
    for members in (("main@v0", "main@v10"), ("main@v20", "main@v30")):
        assert sum(uncertainty.p_best_lineage[e] for e in members) == pytest.approx(1.0)
    (_, second) = fitted.groups
    assert (second.anchor, second.rule) == ("main@v20", "first_trainable")
    assert second.note == "v0 of 'main' did not play here: anchor main@v20"
    assert fitted.groups[0].note is None


def test_a_score_entity_with_one_match_has_no_interval_and_an_infinite_standard_error():
    import json

    table = ComparisonTable()
    for k, (member, score) in enumerate([("a", 1.0), ("a", 3.0), ("b", 2.0)]):
        table.add_result(make_match_result([(0, member, "fixed")], scores=[score], layout="coop", uid=f"c{k}"), [None])
    summary = score_summary(table, "coop", sources=TRAIN)
    assert (summary.entities["b"].ci_low, summary.entities["b"].ci_high) == (None, None)
    assert summary.by_entity["b"] == (1, 2.0, math.inf)
    assert summary.by_entity["a"][2] < math.inf
    assert summary.to_dict()["entities"]["b"]["ci_low"] is None and json.dumps(summary.to_dict())


def test_iterative_fits_of_different_layouts_draw_independently():
    """T4.1 averages contrast samples across layouts draw by draw, so the fits' own draws must be independent
    between layouts (a per-layout seed) and fixed within one (a deterministic fit). Two independent tables as
    layouts "a" and "b": for independent draws the sample correlation of 400 pairs has sd 1 / sqrt(400) = 0.05;
    the bound 0.3 on every one of the 40 versions' |correlation| is 6 sd (two-sided 2e-9 each; shared draws give
    about 0.99): a correct implementation fails with probability below 1e-7."""
    options = BTOptions(dense_limit=5)
    fits = [fit(_chain_table(seed=seed, layout=layout), layout, sources=TRAIN, random_bot="random", options=options)
            for seed, layout in ((11, "a"), (12, "b"))]
    again = fit(_chain_table(seed=11, layout="a"), "a", sources=TRAIN, random_bot="random", options=options)
    assert np.array_equal(again.draws, fits[0].draws)
    first, second = (laplace(f, samples=400, rng=np.random.default_rng(0)) for f in fits)
    correlations = [abs(float(np.corrcoef(first.contrast_samples[e], second.contrast_samples[e])[0, 1]))
                    for e in first.contrast_samples if e != "random"]
    assert len(correlations) == 40 and max(correlations) < 0.3, max(correlations)


def test_solver_shortfalls_are_reported(monkeypatch, caplog):
    """A G2 refit that does not converge is untestable (p None, not flagged); conjugate-gradient solves that stop
    at max_iter (Newton steps, perturb-and-solve draws) are counted in one WARNING each."""
    import functools

    import colosseum.arena.bt as bt

    cycle = _duels({("A", "B"): (240, 60), ("B", "C"): (240, 60), ("C", "A"): (240, 60)})
    fitted = fit(cycle, "2p", sources=TRAIN)
    real_newton = bt._newton
    monkeypatch.setattr(bt, "_newton", lambda *a, **k: (*real_newton(*a, **k)[:2], False, 0))
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.bt"):
        result = transitivity_test(cycle, fitted, sources=TRAIN)
    assert (result.p_value, result.flagged) == (None, False) and result.g2 > 100
    assert "reported as untestable" in caplog.text
    monkeypatch.undo()
    caplog.clear()
    monkeypatch.setattr(bt, "_cg", functools.partial(bt._cg, max_iter=1))
    monkeypatch.setattr(bt, "_block_cg", functools.partial(bt._block_cg, max_iter=2))
    with caplog.at_level(logging.WARNING, logger="colosseum.arena.bt"):
        fit(_chain_table(10), "2p", sources=TRAIN, random_bot="random", options=BTOptions(dense_limit=5))
    assert "Newton steps did not reach the conjugate-gradient tolerance" in caplog.text
    assert f"of {bt.ITERATIVE_SAMPLES} perturb-and-solve draws did not reach" in caplog.text
