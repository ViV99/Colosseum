"""Report data builders (SP4 spec block 5, T7.1): every section for wdl / rank / score runs, size limits,
counter ordering, partial metrics lines, SP3 run dirs without a journal, live vs journal ratings, filters."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from colosseum.arena.report.data import (
    MIN_CELL_GAMES,
    SECTIONS,
    MetricsTail,
    ReportBuilder,
    ReportLimits,
    build_report_data,
    section,
    thin,
)
from game_helpers import copy_sp3_run
from report_kit import VERSIONS, write_run_dir


def _groups(data: dict, layout: str) -> list[dict]:
    return data["sections"]["ratings"]["layouts"][layout]["groups"]


def test_thin_keeps_first_and_last_and_bounds_the_count():
    assert thin(5, 10) == [0, 1, 2, 3, 4]
    idx = thin(10_001, 2000)
    assert len(idx) <= 2000 and idx[0] == 0 and idx[-1] == 10_000 and idx == sorted(set(idx))
    assert thin(7, 1) == [6]


@pytest.mark.parametrize("kind", ["wdl", "rank", "score"])
def test_every_section_is_built_for_each_outcome_kind(tmp_path, kind):
    run = write_run_dir(tmp_path, kind)
    data = build_report_data(run.root)
    assert data["format"] == 1 and set(data["sections"]) == set(SECTIONS)
    json.dumps(data, allow_nan=False)                       # embeddable as JSON
    ratings = data["sections"]["ratings"]
    assert ratings["source"] == "journal" and ratings["uncertainty"] == "bootstrap"
    layout = ratings["layouts"][run.layout]
    matrix = data["sections"]["matrix"]["layouts"][run.layout]
    if kind == "score":
        assert layout["kind"] == "score" and matrix["kind"] == "score"
        assert {r["lineage"] for r in layout["entities"]} == {"main", "greedy", "random"}
        assert layout["compositions"] and all(c["ci"][0] <= c["mean"] <= c["ci"][1] for c in layout["compositions"])
    else:
        (group,) = _groups(data, run.layout)
        names = {r["entity"] for r in group["entities"]}
        assert {"greedy", "random"} <= names and {f"main@v{v}" for v in VERSIONS} <= names
        (series,) = group["series"]
        assert series["lineage"] == "main" and [p[0] for p in series["points"]] == list(VERSIONS)
        assert {a["entity"] for a in group["anchors"]} == {"greedy", "random"}
        assert set(matrix["lineages"]["axis"]) == {"main", "greedy", "random"}
        assert matrix["snapshots"]["agent"] == "main"
        anchors = data["sections"]["anchors"]
        assert anchors["source"] == "train" and anchors["biased"] is True
        assert set(anchors["layouts"][run.layout]["main"]["anchors"]) == {"greedy", "random"}
        assert data["sections"]["opponents"]["agents"]["main"][run.layout]["categories"]["anchors"][0] == 0.3
        assert data["sections"]["opponents"]["pfsp"][run.layout]["main"][0]["player"] == "greedy@fixed"
    summary = data["sections"]["summary"]
    assert summary["status"] == "finished" and summary["env_steps"] == 39000
    assert {a["agent"]: a["kind"] for a in summary["agents"]} == {"main": "trainable", "greedy": "scripted",
                                                                 "random": "scripted"}
    assert data["sections"]["matches"]["rows"] and data["sections"]["tournaments"]["items"] == []


def test_strength_order_survives_into_ratings_and_matrix(tmp_path):
    run = write_run_dir(tmp_path, "wdl", per_pair=60)
    data = build_report_data(run.root)
    rating = {r["entity"]: r["rating"] for r in _groups(data, "2p")[0]["entities"]}
    assert rating["random"] == 0.0                              # the default anchor: the config's RandomBot
    assert rating[f"main@v{VERSIONS[-1]}"] > rating["main@v0"]
    assert rating["greedy"] > rating["random"]
    axis = data["sections"]["matrix"]["layouts"]["2p"]["lineages"]["axis"]
    assert axis.index("greedy") < axis.index("random")          # ordered by rating


def test_limits_bound_every_list_and_order_by_counters(tmp_path):
    run = write_run_dir(tmp_path, "wdl", train_records=300, decreasing_ts=True)
    limits = ReportLimits(series_points=20, entity_rows=4, matches=7, snapshot_matrix=5)
    data = build_report_data(run.root, limits=limits)
    steps = data["sections"]["training"]["agents"]["main"]["train_step"]
    assert len(steps) <= 20 and steps == sorted(steps) and steps[0] == 0 and steps[-1] == 2990
    assert len(data["sections"]["system"]["env_steps"]) <= 20
    (group,) = _groups(data, "2p")
    assert len(group["entities"]) == 4 and group["truncated"] == len(VERSIONS) + 2 - 4
    assert len(data["sections"]["matches"]["rows"]) == 7
    snap = data["sections"]["matrix"]["layouts"]["2p"]["snapshots"]
    assert len(snap["axis"]) <= 5 and snap["thinned_from"] > 5
    cells = data["sections"]["matrix"]["layouts"]["2p"]["lineages"]["cells"]
    assert all(len(row) == 3 for row in cells)


def test_few_games_are_marked_by_the_count_not_dropped(tmp_path):
    run = write_run_dir(tmp_path, "wdl", per_pair=4)
    snap = build_report_data(run.root)["sections"]["matrix"]["layouts"]["2p"]["snapshots"]
    counts = [c["n"] for row in snap["cells"] for c in row if c is not None]
    assert counts and min(counts) < MIN_CELL_GAMES


def test_metrics_tail_waits_for_a_complete_last_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"kind": "train", "train_step": 1}\n{"kind": "train", "train_st')
    tail = MetricsTail(path)
    assert [r["train_step"] for r in tail.poll()] == [1]
    with path.open("a") as fh:
        fh.write('ep": 2}\nnot json\n')
    assert [r["train_step"] for r in tail.poll()] == [1, 2] and tail.bad_lines == 1


def test_a_running_run_with_a_partial_line_is_read(tmp_path):
    run = write_run_dir(tmp_path, "wdl", status="running", partial_last_line=True)
    data = build_report_data(run.root)
    assert data["sections"]["summary"]["status"] == "running"
    assert data["sections"]["training"]["notes"] == []           # the partial line is not "malformed"


def test_live_ratings_come_from_ratings_json_while_running(tmp_path):
    run = write_run_dir(tmp_path, "wdl", status="running", live_bt=True)
    ratings = build_report_data(run.root)["sections"]["ratings"]
    assert ratings["source"] == "live" and ratings["uncertainty"] == "laplace"
    (group,) = ratings["layouts"]["2p"]["groups"]                  # the '@' extras are not groups
    rows = {r["entity"]: r for r in group["entities"]}
    assert rows["main@v200"]["rating"] == 123.0 and rows["main@v200"]["ci"] == [100.0, 146.0]
    assert rows["main@v200"]["p_best_group"] == 0.6 and group["anchor"] == "random" and group["label"] == "player"
    assert ratings["layouts"]["2p"]["bias"] == {"0": 12.0, "1": -12.0}
    assert ratings["layouts"]["2p"]["transitivity"]["flag"] is False


def test_a_dead_running_process_is_stopped_and_rated_from_the_journal(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    run = write_run_dir(tmp_path, "wdl", status="running", pid=proc.pid, live_bt=True)
    data = build_report_data(run.root)
    assert data["sections"]["summary"]["status"] == "stopped"
    assert data["sections"]["ratings"]["source"] == "journal"


def test_pulse_curves_and_unevaluated_snapshots(tmp_path):
    run = write_run_dir(tmp_path, "wdl", evaluation=True, pulse_versions=(160, 200), stored_versions=(160, 180, 200))
    anchors = build_report_data(run.root)["sections"]["anchors"]
    assert anchors["source"] == "pulse" and anchors["biased"] is False
    entry = anchors["layouts"]["2p"]["main"]
    assert [p[0] for p in entry["anchors"]["greedy"]] == [160, 200]
    assert entry["unevaluated"] == [180]


def test_sp3_run_dir_degrades_with_notes(tmp_path):
    root = copy_sp3_run(tmp_path)
    data = build_report_data(root)
    assert set(data["sections"]) == set(SECTIONS)
    assert data["sections"]["summary"]["status"] == "unknown"
    assert data["sections"]["ratings"]["source"] == "none" and data["sections"]["ratings"]["notes"]
    assert data["sections"]["matches"]["rows"] == [] and data["sections"]["matches"]["notes"]
    assert data["sections"]["training"]["agents"]                  # metrics.jsonl of SP3 still drawn
    default = data["sections"]["training"]["default"]              # pre-T3.1 runs: the clipped fractions
    assert default.index("rho_clip_frac") == default.index("rho_trunc_mass") + 1
    json.dumps(data, allow_nan=False)


def test_section_filters_and_unknown_names(tmp_path):
    run = write_run_dir(tmp_path, "wdl", pulse_versions=(200,))
    greedy = section(run.root, "ratings", agent="greedy")
    assert {r["entity"] for r in greedy["layouts"]["2p"]["groups"][0]["entities"]} == {"greedy"}
    pulse = section(run.root, "matches", source="pulse", bogus="x")
    assert pulse["rows"] and {r["source"] for r in pulse["rows"]} == {"pulse"}
    assert pulse["filters"]["source"] == "pulse"
    with pytest.raises(KeyError):
        section(run.root, "nope")


def test_tournament_reports_are_listed(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    (run.root / "tournaments" / "t1").mkdir(parents=True)
    (run.root / "tournaments" / "t1" / "report.json").write_text(json.dumps({"name": "t1", "mode": "gauntlet"}))
    (run.root / "tournaments" / "t2").mkdir()
    out = build_report_data(run.root)["sections"]["tournaments"]
    assert [i["name"] for i in out["items"]] == ["t1"] and out["items"][0]["report"]["mode"] == "gauntlet"
    assert any("t2" in n for n in out["notes"])


def test_fits_are_cached_until_the_journal_changes(tmp_path, monkeypatch):
    from colosseum.arena import bt
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
    from report_kit import INTERVAL, make_results

    run = write_run_dir(tmp_path, "wdl")
    calls = []
    real_fit = bt.fit
    monkeypatch.setattr(bt, "fit", lambda *a, **k: calls.append(1) or real_fit(*a, **k))
    builder = ReportBuilder(run.root)
    try:
        builder.section("ratings")
        builder.section("summary")
        assert len(calls) == 1
        log = MatchLog(run.root / MATCH_LOG_FILE, RunInfo(run_id=run.run_id, name=run.root.name, parent_run_id=None,
                                                          game="turns", config_hash="synthetic",
                                                          checkpoint_interval=INTERVAL))
        try:
            log.append(make_results("wdl", run_id=run.run_id + "-more", per_pair=2, seed=5))
        finally:
            log.close()
        builder.section("ratings")
        assert len(calls) == 2
    finally:
        builder.close()
    assert os.path.exists(run.root / MATCH_LOG_FILE)


def test_null_intervals_never_become_inf_or_nan(tmp_path):
    """A one-game entity of a score layout has no interval (T2.1: ``ci_low`` / ``ci_high`` None, se inf): the
    report carries ``null``, never ``inf`` / ``nan``."""
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
    from colosseum.core.types import MatchResult, SeatResult, TeamResult
    from report_kit import INTERVAL

    run = write_run_dir(tmp_path, "score")
    once = MatchResult(match_id="x", layout="solo", outcome_kind="score",
                       seats=[SeatResult(0, "player", 0, "greedy2", "fixed", 0.0)], teams=[TeamResult(0, 1.0, 2.5)],
                       episode_length=3, uid=f"{run.run_id}/train/once", source="train", seed=1, env_steps=1)
    log = MatchLog(run.root / MATCH_LOG_FILE, RunInfo(run_id=run.run_id, name=run.root.name, parent_run_id=None,
                                                      game="solo", config_hash="synthetic",
                                                      checkpoint_interval=INTERVAL))
    try:
        log.append([once])
    finally:
        log.close()
    data = build_report_data(run.root)
    rows = {r["entity"]: r for r in data["sections"]["ratings"]["layouts"]["solo"]["entities"]}
    assert rows["greedy2"]["n"] == 1 and rows["greedy2"]["ci"] == [None, None]
    cross = data["sections"]["matrix"]["layouts"]["solo"]["cross_play"]
    assert next(c for c in cross if c["composition"] == "greedy2")["ci"] == [None, None]
    json.dumps(data, allow_nan=False)


def test_a_bootstrap_that_does_not_match_the_table_falls_back_to_laplace(tmp_path, monkeypatch):
    from colosseum.arena import bt

    run = write_run_dir(tmp_path, "wdl")

    def mismatch(*args, **kwargs):
        raise ValueError("bootstrap data has a team pair the fit does not have")

    monkeypatch.setattr(bt, "bootstrap", mismatch)
    ratings = build_report_data(run.root)["sections"]["ratings"]
    assert ratings["uncertainty"] == "laplace" and any("Laplace" in n for n in ratings["notes"])
    (group,) = ratings["layouts"]["2p"]["groups"]
    assert all(r["ci"][0] is not None for r in group["entities"] if r["entity"] != "random")


def test_a_running_score_run_without_a_journal_reads_the_live_scores(tmp_path):
    run = write_run_dir(tmp_path, "score", status="running", journal=False)
    ratings = json.loads((run.root / "ratings.json").read_text())
    ratings["bt"] = {"solo": {"@score": {
        "entities": [{"entity": "main@v200", "n": 30, "mean": 3.5, "ci_low": 3.2, "ci_high": 3.8},
                     {"entity": "random", "n": 1, "mean": 2.0, "ci_low": None, "ci_high": None}],
        "compositions": [{"composition": "main@v200", "n": 30, "mean": 3.5, "ci_low": 3.2, "ci_high": 3.8,
                          "homogeneous": True}]}}}
    (run.root / "ratings.json").write_text(json.dumps(ratings))
    section_ = build_report_data(run.root)["sections"]["ratings"]
    assert section_["source"] == "live"
    layout = section_["layouts"]["solo"]
    assert [r["entity"] for r in layout["entities"]] == ["main@v200", "random"]
    assert layout["entities"][1]["ci"] == [None, None] and layout["compositions"][0]["homogeneous"] is True


def test_unreadable_tournament_reports_are_notes(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    for name, text in (("bad", "{not json"), ("list", "[1, 2]"), ("broken", json.dumps({"tables": [{"x": 1}]}))):
        (run.root / "tournaments" / name).mkdir(parents=True)
        (run.root / "tournaments" / name / "report.json").write_text(text)
    out = build_report_data(run.root)["sections"]["tournaments"]
    assert out["items"] == [] and len(out["notes"]) == 3


def test_the_summary_states_the_retention_rule(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    summary = build_report_data(run.root)["sections"]["summary"]
    assert summary["retention"].startswith("snapshots: newest")
