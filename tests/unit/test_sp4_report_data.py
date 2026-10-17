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


def _append_more(run, *, per_pair: int = 2, seed: int = 5, tag: str = "-more") -> None:
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo
    from report_kit import INTERVAL, make_results

    log = MatchLog(run.root / MATCH_LOG_FILE, RunInfo(run_id=run.run_id, name=run.root.name, parent_run_id=None,
                                                      game="turns", config_hash="synthetic",
                                                      checkpoint_interval=INTERVAL))
    try:
        log.append(make_results("wdl", run_id=run.run_id + tag, per_pair=per_pair, seed=seed))
    finally:
        log.close()


def test_a_write_between_reads_keeps_fit_bootstrap_and_test_consistent(tmp_path, monkeypatch):
    """The race of a live journal: a write lands right after the bootstrap read and while the builder holds an
    older cached table (dashboard, ``table_min_age_sec``). The fit, its bootstrap intervals and the G2 test come
    from that one read (``BootstrapData.to_table``), so they agree; the next build refits on the new journal."""
    from colosseum.arena import bt
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog

    run = write_run_dir(tmp_path, "wdl")
    builder = ReportBuilder(run.root, table_min_age_sec=30.0)
    real = bt.bootstrap_data
    seen = []

    def read_then_write(reader, layout, *, sources):
        data = real(reader, layout, sources=sources)
        seen.append(data)
        if len(seen) == 1:
            _append_more(run)                                 # lands after the read, before fit / bootstrap
        return data

    monkeypatch.setattr(bt, "bootstrap_data", read_then_write)
    try:
        assert builder.section("anchors")["source"] == "train"     # caches the RUN_SOURCES table first
        _append_more(run, seed=6, tag="-early")                  # newer than the cached table
        ratings = builder.section("ratings")
        assert ratings["uncertainty"] == "bootstrap" and ratings["notes"] == []
        (group,) = ratings["layouts"]["2p"]["groups"]
        games = {r["entity"]: r["games"] for r in group["entities"]}
        expected = seen[0].to_table("2p").entity_games("2p", (bt.DATA_SOURCE,))
        assert games == {e: n for e, n in expected.items() if e in games}
        assert all(r["ci"][0] <= r["rating"] <= r["ci"][1] for r in group["entities"] if r["entity"] != "random")
    finally:
        builder.close()
    fresh = ReportBuilder(run.root)                             # the next build sees every match
    try:
        (group,) = fresh.section("ratings")["layouts"]["2p"]["groups"]
        with MatchLog.open_readonly(run.root / MATCH_LOG_FILE) as reader:
            journal = reader.comparison_table().entity_games("2p", ("train", "pulse"))
        assert {r["entity"]: r["games"] for r in group["entities"]} == {
            e: n for e, n in journal.items() if e in {r["entity"] for r in group["entities"]}}
        assert len(seen) == 2
    finally:
        fresh.close()


def test_the_metrics_tail_keeps_only_the_report_kinds(tmp_path):
    path = tmp_path / "metrics.jsonl"
    lines = [{"kind": "train", "train_step": 1}, {"kind": "ratings", "layouts": {"big": list(range(100))}},
             {"kind": "system", "env_steps": 1}, {"kind": "episodes", "env_steps": 1}, {"kind": "custom"}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    tail = MetricsTail(path)
    assert [r["kind"] for r in tail.poll()] == ["train", "system", "episodes"]
    assert tail.of("ratings") == [] and len(tail.of("train")) == 1
    assert len(MetricsTail(path, kinds=None).poll()) == 5


def test_the_metrics_tail_thins_each_series_by_its_counter(tmp_path):
    """Bounded memory over a run of any length (ruling C-M2): each (kind, agent) series is thinned on ingest by its
    counter (stride doubling) to at most ``max_per_series`` records plus the newest; the first and the newest
    record stay, the kept counters are sorted and evenly spaced, and every series is thinned on its own."""
    path = tmp_path / "metrics.jsonl"
    path.write_text("")
    cap = 50
    tail = MetricsTail(path, max_per_series=cap)
    for batch in range(20):                               # many polls, as a long run's dashboard makes them
        with path.open("a") as fh:
            for i in range(500):
                k = batch * 500 + i
                fh.write(json.dumps({"kind": "train", "agent": "main", "train_step": 10 * k}) + "\n")
                if k % 5 == 0:
                    fh.write(json.dumps({"kind": "train", "agent": "other", "train_step": k}) + "\n")
                fh.write(json.dumps({"kind": "system", "env_steps": 100 * k}) + "\n")
        tail.poll()
        assert len(tail.of("train")) <= 2 * (cap + 1) and len(tail.of("system")) <= cap + 1
    main = [r["train_step"] for r in tail.of("train") if r["agent"] == "main"]
    other = [r["train_step"] for r in tail.of("train") if r["agent"] == "other"]
    system = [r["env_steps"] for r in tail.of("system")]
    for series, last in ((main, 99_990), (other, 9_995), (system, 999_900)):
        assert cap // 2 <= len(series) <= cap + 1
        assert series == sorted(series) and series[0] == 0 and series[-1] == last
        gaps = [b - a for a, b in zip(series, series[1:])]
        assert max(gaps) <= 2 * (series[-1] - series[0]) / (len(series) - 1)      # evenly spaced


def test_the_report_builder_bounds_its_metrics_tail_by_the_page_limits(tmp_path):
    """The builder's tail keeps 2 x ``series_points`` records per series (so the page still gets its
    ``series_points`` after a halving), not every record of the run."""
    run = write_run_dir(tmp_path, "wdl", train_records=600)
    builder = ReportBuilder(run.root, limits=ReportLimits(series_points=10))
    try:
        steps = builder.section("training")["agents"]["main"]["train_step"]
        assert len(builder.records("train")) <= 2 * 10 + 1
        assert len(steps) == 10 and steps[0] == 0 and steps[-1] == 5990
    finally:
        builder.close()


def test_the_metrics_tail_reads_in_fixed_size_chunks(tmp_path, monkeypatch):
    """Chunked reads give the same records as one read: lines across chunk borders, a line longer than a chunk, a
    malformed line and a partial last line that completes later."""
    import colosseum.arena.report.data as data_mod

    path = tmp_path / "metrics.jsonl"
    lines = [json.dumps({"kind": "train", "agent": "main", "train_step": i, "pad": "x" * (i % 7) * 10})
             for i in range(40)]
    path.write_text("\n".join(lines) + "\nnot json\n" + '{"kind": "train", "agent": "main", "train_st')
    whole = MetricsTail(path)
    assert [r["train_step"] for r in whole.poll()] == list(range(40)) and whole.bad_lines == 1
    monkeypatch.setattr(data_mod, "READ_CHUNK_BYTES", 7)
    tail = MetricsTail(path)
    assert [r["train_step"] for r in tail.poll()] == list(range(40)) and tail.bad_lines == 1
    assert tail.offset == whole.offset == path.stat().st_size - len('{"kind": "train", "agent": "main", "train_st')
    with path.open("a") as fh:
        fh.write('ep": 40}\n')
    assert [r["train_step"] for r in tail.poll()] == list(range(41)) and tail.offset == path.stat().st_size


def test_ratings_json_is_parsed_once_per_change(tmp_path, monkeypatch):
    import colosseum.arena.report.data as data_mod

    run = write_run_dir(tmp_path, "wdl")
    calls = []
    real = data_mod.json.loads
    monkeypatch.setattr(data_mod.json, "loads", lambda text, *a, **k: calls.append(1) or real(text, *a, **k))
    builder = ReportBuilder(run.root)
    try:
        builder.ratings_json()
        builder.ratings_json()
        assert len(calls) == 1
        path = run.root / "ratings.json"
        path.write_text(json.dumps({"env_steps": 1, "layouts": {}}) + " " * 7)
        assert builder.ratings_json()["env_steps"] == 1 and len(calls) == 2
    finally:
        builder.close()


def test_score_compositions_are_capped_and_filtered_by_agent(tmp_path):
    run = write_run_dir(tmp_path, "score")
    limits = ReportLimits(entity_rows=2)
    data = build_report_data(run.root, limits=limits)
    layout = data["sections"]["ratings"]["layouts"]["solo"]
    assert len(layout["entities"]) == 2 and layout["truncated"] == len(VERSIONS) + 2 - 2
    total = len(VERSIONS) + 2                                     # one-seat teams: one composition per entity
    assert len(layout["compositions"]) == 2 and layout["compositions_truncated"] == total - 2
    cross = data["sections"]["matrix"]["layouts"]["solo"]
    assert len(cross["cross_play"]) == 2 and cross["truncated"] == total - 2
    greedy = section(run.root, "ratings", agent="greedy")["layouts"]["solo"]
    assert [c["composition"] for c in greedy["compositions"]] == ["greedy"]
    assert [e["entity"] for e in greedy["entities"]] == ["greedy"]


def test_pulse_anchors_of_a_score_game_and_agents_without_pulse_results(tmp_path):
    run = write_run_dir(tmp_path, "score", evaluation=True, pulse_versions=(160, 200))
    anchors = build_report_data(run.root)["sections"]["anchors"]
    assert anchors["source"] == "pulse" and not any("no results" in n for n in anchors["notes"])
    entry = anchors["layouts"]["solo"]["main"]
    assert entry["metric"] == "score" and [p[0] for p in entry["anchors"]["(self)"]] == [160, 200]
    assert entry["unevaluated"] == [180] and anchors["unevaluated"] == {"main": [180]}
    json.dumps(anchors, allow_nan=False)
    other = write_run_dir(tmp_path, "wdl", name="no-pulse-yet", evaluation=True)
    anchors = build_report_data(other.root)["sections"]["anchors"]
    assert anchors["layouts"] == {} and anchors["unevaluated"] == {"main": [160, 180, 200]}
    assert any("no results" in n for n in anchors["notes"])


def test_an_empty_evaluation_agent_list_marks_no_snapshot_unevaluated(tmp_path):
    """``evaluation.agents: []`` evaluates no agent (as ``pulse_plan`` reads it; null = every trainable agent),
    so no stored snapshot is listed as "unevaluated"."""
    import yaml

    run = write_run_dir(tmp_path, "wdl", evaluation=True)
    path = run.root / "config.resolved.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["evaluation"]["agents"] = []
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    anchors = build_report_data(run.root)["sections"]["anchors"]
    assert anchors["source"] == "pulse" and anchors["unevaluated"] == {}


@pytest.mark.parametrize("content", [None, "agents: [unterminated"])
def test_a_missing_or_unreadable_config_is_a_note(tmp_path, content):
    run = write_run_dir(tmp_path, "wdl")
    path = run.root / "config.resolved.yaml"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    data = build_report_data(run.root)
    notes = data["sections"]["summary"]["notes"]
    assert any("config.resolved.yaml" in n for n in notes)
    assert any("config.resolved.yaml" in n for n in data["sections"]["ratings"]["notes"])
    json.dumps(data, allow_nan=False)


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
