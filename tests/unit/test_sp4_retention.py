"""Snapshot retention, the pure function (SP4 T4.1, spec block 4): table tests of every rule, the cap order,
and the combination of per-layout ratings into one score per snapshot."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from colosseum.arena.retention import (
    RetentionPolicy,
    SnapshotInfo,
    SnapshotScore,
    combine_scores,
    describe_retention,
    retain,
)
from colosseum.core.config import CheckpointConfig

RUN = "run-now"


def snaps(*versions, final=(), pinned=(), run_id=RUN) -> list[SnapshotInfo]:
    return [SnapshotInfo(f"ckpt_v{v}", v, v in final, run_id, v in pinned) for v in versions]


def policy(**kw) -> RetentionPolicy:
    base = dict(keep_last=2, keep_every_period=0, keep_top_k=0, top_k_min_games=0, max_snapshots=None,
                current_run_id=RUN)
    return RetentionPolicy(**{**base, **kw})


def kept(decision) -> list[int]:
    return [int(c.removeprefix("ckpt_v")) for c in decision.keep]


def scored(**lcbs) -> dict[str, SnapshotScore]:
    """``scored(v10=(lcb, games))``."""
    return {f"ckpt_{k}": SnapshotScore(lcb, games) for k, (lcb, games) in lcbs.items()}


def test_keep_last_keep_every_and_the_snapshot_just_saved():
    d = retain(snaps(10, 20, 30, 40, 50, 60), {}, policy(keep_last=2, keep_every_period=20), protect="ckpt_v10")
    assert kept(d) == [10, 20, 40, 50, 60]
    assert d.evict == ["ckpt_v30"] and d.reasons["ckpt_v30"].startswith("evicted")
    assert "just saved" in d.reasons["ckpt_v10"] and "keep_every" in d.reasons["ckpt_v20"]
    assert d.warnings == []


def test_only_the_final_snapshot_of_the_current_run_is_protected():
    entries = [SnapshotInfo("ckpt_v7", 7, True, "parent", False), SnapshotInfo("ckpt_v9", 9, True, None, False),
               SnapshotInfo("ckpt_v11", 11, True, RUN, False), *snaps(20, 30)]
    assert kept(retain(entries, {}, policy())) == [11, 20, 30]
    # a manager without a run id (tests, SP3 callers) keeps the SP3 rule: every final snapshot is protected
    assert kept(retain(entries, {}, policy(current_run_id=None))) == [7, 9, 11, 20, 30]


def test_pinned_snapshots_are_never_evicted_not_even_by_the_cap():
    d = retain(snaps(10, 20, 30, 40, pinned=(10, 20)), {}, policy(keep_last=2, max_snapshots=3))
    assert kept(d) == [10, 20, 30, 40]
    assert len(d.warnings) == 1 and "max_snapshots=3" in d.warnings[0]


def test_top_k_by_lcb_among_all_rated_snapshots_ties_go_to_the_newer():
    scores = scored(v10=(5.0, 500), v20=(9.0, 500), v30=(7.0, 500), v40=(7.0, 500), v50=(1.0, 500), v60=(0.0, 500))
    d = retain(snaps(10, 20, 30, 40, 50, 60), scores, policy(keep_last=1, keep_top_k=2, top_k_min_games=100))
    assert kept(d) == [20, 40, 60]
    assert "top_k" in d.reasons["ckpt_v20"]


def test_new_snapshots_are_protected_until_they_have_enough_rated_games():
    scores = scored(v10=(9.0, 500), v20=(1.0, 50), v30=(0.5, 500))
    d = retain(snaps(10, 20, 30, 40, 50), scores, policy(keep_last=2, keep_top_k=1, top_k_min_games=100))
    assert kept(d) == [10, 20, 40, 50]          # 20: 50 < 100 games; 30: rated, not top-1
    assert "new" in d.reasons["ckpt_v20"]
    unrated = retain(snaps(10, 20, 30), {}, policy(keep_last=1, keep_top_k=1, top_k_min_games=1))
    assert kept(unrated) == [10, 20, 30]        # no scores: everything is new
    off = retain(snaps(10, 20, 30), {}, policy(keep_last=1, keep_top_k=1, top_k_min_games=0))
    assert kept(off) == [30]


def test_the_cap_without_top_k_evicts_keep_every_snapshots_oldest_first():
    d = retain(snaps(10, 20, 30, 40, 50, 60), {}, policy(keep_last=2, keep_every_period=10, max_snapshots=4))
    assert kept(d) == [30, 40, 50, 60]
    assert d.reasons["ckpt_v10"] == "evicted: over max_snapshots=4 (keep_every)"


def test_the_cap_with_top_k_order_keep_every_new_top_k():
    entries = snaps(10, 20, 30, 40, 50, 60, 70)
    scores = scored(v10=(3.0, 500), v20=(8.0, 500), v30=(1.0, 10), v40=(2.0, 500), v50=(None, 500))
    pol = policy(keep_last=1, keep_every_period=10, keep_top_k=2, top_k_min_games=100)
    # top-k: 20 (8.0), 10 (3.0); new (< 100 games, outside keep_last): 30, 60 (no score); keep_every-only: 40, 50.
    # The cap evicts the unrated keep_every-only 50, then 40 (by lcb), then the new 30, 60 (oldest first), then
    # the top-k 10 (lower lcb) and 20; 70 (keep_last) stays.
    order = []
    for cap in (6, 5, 4, 3, 2, 1):
        d = retain(entries, scores, policy(**{**pol.__dict__, "max_snapshots": cap}))
        order.append(kept(d))
    assert order == [[10, 20, 30, 40, 60, 70], [10, 20, 30, 60, 70], [10, 20, 60, 70], [10, 20, 70], [20, 70], [70]]


def test_bad_policies_and_duplicate_ids_are_value_errors():
    with pytest.raises(ValueError):
        retain(snaps(1), {}, policy(keep_last=0))
    with pytest.raises(ValueError):
        retain(snaps(1), {}, policy(max_snapshots=0))
    with pytest.raises(ValueError, match="duplicate"):
        retain(snaps(1, 1), {}, policy())


def test_entries_in_any_order_give_ids_by_version():
    d = retain(list(reversed(snaps(10, 20, 30))), {}, policy(keep_last=2))
    assert d.keep == ["ckpt_v20", "ckpt_v30"] and d.evict == ["ckpt_v10"]


@dataclass
class _Fit:            # the BTFit fields combine_scores reads
    rating: dict
    games: dict


@dataclass
class _Unc:            # the Uncertainty fields combine_scores reads
    lcb: dict
    contrast_samples: dict = field(default_factory=dict)


@dataclass
class _Scores:         # the ScoreSummary field combine_scores reads
    by_entity: dict


def test_combine_scores_weights_the_draws_of_every_layout():
    rng = np.random.default_rng(0)
    a, b = rng.normal(100.0, 10.0, 4000), rng.normal(0.0, 10.0, 4000)
    fits = {"2p": _Fit({"main@v10": 100.0, "main@v20": 1.0, "greedy": 0.0}, {"main@v10": 300, "main@v20": 7}),
            "4p": _Fit({"main@v10": 0.0}, {"main@v10": 50})}
    unc = {"2p": _Unc({"main@v10": 87.0, "main@v20": -5.0}, {"main@v10": a}), "4p": _Unc({"main@v10": -13.0},
                                                                                       {"main@v10": b})}
    out = combine_scores("main", fits=fits, uncertainty=unc, scores={}, layout_weights={"2p": 3.0, "4p": 1.0},
                         quantile=0.1)
    assert set(out) == {"ckpt_v10", "ckpt_v20"} and "greedy" not in out
    assert out["ckpt_v10"].games == 350 and out["ckpt_v20"] == SnapshotScore(-5.0, 7)   # one layout: its lcb
    expected = np.quantile(0.75 * a + 0.25 * b, 0.1)
    assert out["ckpt_v10"].lcb == pytest.approx(expected)
    other = combine_scores("other", fits=fits, uncertainty=unc, scores={}, layout_weights={}, quantile=0.1)
    assert other == {}


def test_combine_scores_of_score_layouts_uses_mean_and_standard_error():
    scores = {"coop2": _Scores({"main@v10": (100, 2.0, 0.1), "helper": (5, 1.0, 1.0)})}
    out = combine_scores("main", fits={}, uncertainty={}, scores=scores, layout_weights={"coop2": 1.0},
                         quantile=0.1)
    assert out["ckpt_v10"].games == 100
    assert out["ckpt_v10"].lcb == pytest.approx(2.0 - 1.2815515655446004 * 0.1)


def test_describe_retention_names_every_rule():
    text = describe_retention(CheckpointConfig(interval=1000, keep_last=20, keep_every=10))
    assert "newest 20" in text and "every 10000th version" in text and "top-k off" in text and "at most 100" in text
    top = describe_retention(CheckpointConfig(keep_top_k=5, top_k_min_games=200, max_snapshots=None))
    assert "top 5" in top and "200 rated games" in top and "no cap" in top


def test_combine_scores_truncates_draws_of_different_lengths_and_handles_infinite_errors():
    rng = np.random.default_rng(1)
    a, b = rng.normal(10.0, 1.0, 2000), rng.normal(-10.0, 1.0, 400)      # Laplace vs iterative draw counts
    fits = {"2p": _Fit({"main@v10": 0.0}, {"main@v10": 30}), "4p": _Fit({"main@v10": 0.0}, {"main@v10": 90})}
    unc = {"2p": _Unc({"main@v10": 9.0}, {"main@v10": a}), "4p": _Unc({"main@v10": -11.0}, {"main@v10": b})}
    out = combine_scores("main", fits=fits, uncertainty=unc, scores={}, layout_weights={"2p": 1.0, "4p": 1.0},
                         quantile=0.1)
    assert out["ckpt_v10"].lcb == pytest.approx(np.quantile(0.5 * a[:400] + 0.5 * b, 0.1))
    # a score-only entity with one match has se = inf: never certain, so its lower bound is -inf (no NaN, also
    # next to a zero-weight layout)
    scores = {"coop2": _Scores({"main@v10": (1, 5.0, float("inf")), "main@v20": (9, 1.0, 0.5)}),
              "coop3": _Scores({"main@v20": (4, 0.0, float("inf"))})}
    out = combine_scores("main", fits={}, uncertainty={}, scores=scores,
                         layout_weights={"coop2": 1.0, "coop3": 0.0}, quantile=0.1)
    assert out["ckpt_v10"] == SnapshotScore(float("-inf"), 1)
    assert out["ckpt_v20"].lcb == pytest.approx(1.0 - 1.2815515655446004 * 0.5)
    assert out["ckpt_v20"].games == 13          # games count every layout, the lower bound only the weighted ones
    d = retain(snaps(10, 20, 30), out, policy(keep_last=1, keep_top_k=1, top_k_min_games=0))
    assert kept(d) == [20, 30]                  # -inf is rated, and the lowest


def test_no_ratings_yet_is_not_rated_with_zero_games():
    entries = snaps(10, 20, 30, 40, 50)
    pol = policy(keep_last=1, keep_top_k=1, top_k_min_games=0, max_snapshots=2)
    rated_none = retain(entries, {}, pol)                   # rated, none of them has a score: rule-based eviction
    assert kept(rated_none) == [50]
    waiting = retain(entries, None, pol)                    # no ratings yet: every unrated one is new, cap 2 x 2
    assert kept(waiting) == [20, 30, 40, 50] and waiting.reasons["ckpt_v20"] == "kept: new (no ratings yet)"
    assert waiting.reasons["ckpt_v10"] == "evicted: over max_snapshots=4 (new)"
    assert kept(retain(entries, None, policy(keep_last=1, keep_top_k=0))) == [50]   # no top-k: scores never matter


def test_combine_scores_refuses_a_lower_bound_without_draws_in_a_counted_layout():
    fits = {"2p": _Fit({"main@v10": 0.0}, {"main@v10": 3}), "4p": _Fit({"main@v10": 0.0}, {"main@v10": 3})}
    unc = {"2p": _Unc({"main@v10": 1.0}, {"main@v10": np.zeros(10)}), "4p": _Unc({"main@v10": 2.0})}
    with pytest.raises(ValueError, match="no contrast_samples in layouts \\['4p'\\]"):
        combine_scores("main", fits=fits, uncertainty=unc, scores={}, layout_weights={}, quantile=0.1)
    # a zero-weight layout is not counted, so its missing draws do not matter
    out = combine_scores("main", fits=fits, uncertainty=unc, scores={}, layout_weights={"2p": 1.0}, quantile=0.1)
    assert out["ckpt_v10"].lcb == 1.0
