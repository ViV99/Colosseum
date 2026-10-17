"""HTML renderer (SP4 spec block 5, T7.2): one self-contained file, data embedded as JSON, every section's
container present, no external resources, script-safe escaping; the page draws every section (headless
Chrome when available) and ``page.js`` is valid JavaScript (node when available)."""
from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import subprocess

import pytest

from colosseum.arena.report.data import SECTIONS
from colosseum.arena.report.render import (
    REPORT_FILE,
    asset,
    embed_json,
    is_run_dir,
    is_tournament_dir,
    render_html,
    write_report,
    write_tournament_html,
)
from colosseum.core.errors import ConfigError
from report_kit import sample_report_data, write_run_dir

ASSETS = ("page.html", "page.css", "page.js")
URL = re.compile(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CHROME = next((shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser") if shutil.which(n)), None)


def _data_of(page: str) -> dict:
    m = re.search(r'<script type="application/json" id="colosseum-data">(.*?)</script>', page, re.S)
    return json.loads(m.group(1))


def _dom(tmp_path, data: dict) -> str:
    page = tmp_path / "report.html"
    page.write_text(render_html(data))
    proc = subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--dump-dom",
                           "--virtual-time-budget=3000", page.as_uri()], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0 or not proc.stdout.strip():
        pytest.fail(f"headless Chrome failed (exit code {proc.returncode}): {proc.stderr[-2000:]}")
    return proc.stdout


def _visible(dom: str) -> str:
    """The drawn page without the embedded scripts (the data JSON would match every string)."""
    return re.sub(r"<[^>]+>", " ", dom.split('<script type="application/json"')[0])


def test_page_embeds_the_data_and_has_every_section():
    data = sample_report_data()
    page = render_html(data)
    assert page.startswith("<!doctype html>") and "{{" not in page
    for name in SECTIONS:
        assert f'id="sec-{name}"' in page
    assert _data_of(page) == json.loads(json.dumps(data))
    config = json.loads(re.search(r'id="colosseum-config">(.*?)</script>', page, re.S).group(1))
    assert config == {"live": False, "refresh_sec": 10}
    assert json.loads(re.search(r'id="colosseum-config">(.*?)</script>', render_html(data, live=True, refresh_sec=3),
                                re.S).group(1)) == {"live": True, "refresh_sec": 3}


def test_no_external_resources_in_assets_or_page():
    for name in ASSETS:
        text = asset(name)
        assert not URL.search(text), name
        assert "@import" not in text and "<link" not in text and "src=" not in text, name
    page = render_html(sample_report_data())
    assert not URL.search(page) and "<script src" not in page and "<link" not in page


def test_strings_in_the_data_cannot_break_out_of_the_script_element():
    data = sample_report_data()
    evil = "</script><script>alert(1)</script>&\u2028"
    data["title"] = evil
    data["sections"]["summary"]["agents"][0]["agent"] = evil
    page = render_html(data)
    assert "<script>alert(1)" not in page
    assert _data_of(page)["sections"]["summary"]["agents"][0]["agent"] == evil
    assert "\\u003c/script" in embed_json({"x": "</script>"})


def test_placeholders_inside_the_data_are_not_substituted():
    data = sample_report_data()
    data["sections"]["summary"]["notes"] = ["{{JS}} {{DATA}}"]
    assert _data_of(render_html(data))["sections"]["summary"]["notes"] == ["{{JS}} {{DATA}}"]


def test_non_finite_numbers_become_null():
    assert json.loads(embed_json({"a": float("nan"), "b": [float("inf"), 1.5], "c": (1, 2)})) == \
        {"a": None, "b": [None, 1.5], "c": [1, 2]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_page_js_is_valid_javascript(tmp_path):
    path = tmp_path / "page.js"
    path.write_text(asset("page.js"))
    assert subprocess.run(["node", "--check", str(path)], capture_output=True, text=True).returncode == 0


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_every_section_is_drawn_in_a_browser(tmp_path):
    dom = _dom(tmp_path, sample_report_data())
    assert 'data-ready="1"' in dom
    assert dom.count('data-rendered="1"') == len(SECTIONS)
    assert "could not be drawn:" not in dom.split("<script")[0]
    assert dom.count("<svg") >= 6                       # rating, anchors, opponents, training, system, matrices
    text = _visible(dom)
    assert "confirmed" in text and "not confirmed" not in text
    assert "snapshots: newest 5" in text                # summary.retention
    # the LCB column is on the lineage-contrast scale, not the BT one next to it: its header says so
    lcb_title = "lower bound of the deviation from the lineage mean; used by top-k"
    assert f'<th class="num" scope="col" title="{lcb_title}">LCB' in dom


def _hard_data() -> dict:
    """The sample with every null and mark the producers can emit: a stopped run, intervals of one-game score
    cells, an untestable G2 test, score-layout anchors with unevaluated snapshots, truncated score lists, and a
    select tournament whose lone finalist is not confirmed, with REPEATED marks, candidate weights, a combined
    table, dropped compositions and self-excluded candidates."""
    data = copy.deepcopy(sample_report_data())
    s = data["sections"]
    s["summary"]["status"], s["summary"]["status_detail"] = "stopped", "the pid is gone"
    s["summary"]["agents"][0]["bt"]["ci"] = [None, None]
    rl = s["ratings"]["layouts"]
    rl["4p"]["transitivity"] = {"flag": False, "g2": 0.0, "dof": 0, "p_value": None, "pairs_tested": 0,
                                "worst_pairs": []}
    rl["solo"]["entities"].append({"entity": "random", "lineage": "random", "version": None, "kind": "scripted",
                                   "n": 1, "mean": 0.5, "ci": [None, None]})
    rl["solo"]["truncated"], rl["solo"]["compositions_truncated"] = 4, 2
    s["matrix"]["layouts"]["solo"]["truncated"] = 3
    # a section draws its first layout (a select switches): the score layout first in the matrix, the
    # untestable rank layout first in the ratings (``_score_first`` puts the score layout first there)
    s["matrix"]["layouts"] = {k: s["matrix"]["layouts"][k] for k in ("solo", "2p")}
    s["ratings"]["layouts"] = {k: rl[k] for k in ("4p", "solo", "2p")}
    s["anchors"] = {"source": "pulse", "biased": False, "notes": [], "unevaluated": {"main": [300], "second": [100]},
                    "layouts": {"solo": {"main": {"metric": "score", "unevaluated": [300],
                                                  "anchors": {"(self)": [[100, 2.5, None, None, 1],
                                                                         [200, 3.0, 2.5, 3.5, 6]],
                                                              "random": [[100, None, None, None, 0]]}}}}}
    t = s["tournaments"]["items"][0]["report"]
    t["confirmed"] = False
    lay = t["layouts"]["2p"]
    lay["dropped"] = [{"candidate": "main@v1500", "composition": ["greedy", "random"], "reason": "no seat"}]
    lay["self_excluded"] = ["greedy"]
    g = lay["groups"]["default"]
    g["candidates"][0]["repeated"] = ["greedy"]
    g["candidates"][0]["score_ci"] = None
    g["matrix"]["main@v2000"]["greedy"].update(ci=None, repeated=True)
    g["select"] = {"winner": "main@v2000", "runner_up": None, "confirmed": False, "diff": None, "diff_ci": None,
                   "indistinguishable": [], "repeated_columns": [], "diff_repeated": [["2p", "greedy"]],
                   "note": "a single finalist: there is no difference to confirm"}
    g["transitivity"] = {"flag": False, "g2": 0.0, "dof": 0, "p_value": None}
    g["candidate_weights"] = {"greedy": [0.0, 1.0]}
    g["combined"] = [{"name": "main@v2000", "score": 0.7, "score_ci": [0.6, 0.8], "repeated": [["2p", "greedy"]]}]
    return data


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_nulls_marks_and_an_unconfirmed_winner_are_drawn(tmp_path):
    dom = _dom(tmp_path, _hard_data())
    assert dom.count('data-rendered="1"') == len(SECTIONS)
    assert "could not be drawn:" not in dom.split("<script")[0]
    text = _visible(dom)
    assert "stopped" in text
    assert text.count("untestable") >= 2                 # null G2 p-values (ratings and tournament)
    assert "not confirmed" in text and "(confirmed)" not in text
    assert "a single finalist" in text
    assert text.count("REPEATED") >= 4                   # row, matrix cell, combined row, difference to runner-up
    assert "no seat" in text and "greedy" in text        # dropped composition, self-excluded candidate
    assert "Not evaluated" in text and "v300" in text and "second" in text
    assert "3 more compositions" in text and "main@v100+random" in text       # score cross-play (matrix)
    text = _visible(_dom(tmp_path, _score_first(_hard_data())))
    assert "4 more entities" in text and "2 more compositions" in text        # score layout (ratings)
    assert "main@v100+random" in text and "mean score" in text


def _score_first(data: dict) -> dict:
    layouts = data["sections"]["ratings"]["layouts"]
    data["sections"]["ratings"]["layouts"] = {"solo": layouts.pop("solo"), **layouts}
    return data


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_replay_links_only_for_allowed_protocols(tmp_path):
    """A replay link is a link only for http(s) / file URLs; every other string is plain text. The bypasses of a
    ``javascript:`` deny-list (a tab inside the scheme, a leading control character) stay text."""
    data = sample_report_data()
    rows = data["sections"]["matches"]["rows"]
    replays = ["https://replays.test/r/1", "file:///data/replays/r2.html", "java\tscript:alert(1)",
               "\x01javascript:alert(2)", " javascript:alert(3)", "JaVaScRiPt:alert(4)", "data:text/html,<b>x</b>",
               "vbscript:msgbox(5)"]
    for row, replay in zip(rows, replays):
        row["replay"] = replay
    dom = _dom(tmp_path, data)
    hrefs = [x for x in re.findall(r'<a href="([^"]*)"', dom) if not x.startswith("#")]     # not the section menu
    assert sorted(hrefs) == ["file:///data/replays/r2.html", "https://replays.test/r/1"]
    text = _visible(dom)
    assert "alert(1)" in text and "alert(2)" in text and "alert(4)" in text and "vbscript:msgbox(5)" in text


def test_write_report_default_path_and_output(tmp_path):
    run = write_run_dir(tmp_path, "rank")
    path = write_report(run.root)
    assert path == run.root / REPORT_FILE and _data_of(path.read_text())["sections"]["ratings"]["layouts"]
    other = write_report(run.root, tmp_path / "x.html")
    assert other == tmp_path / "x.html" and other.exists()
    (tmp_path / "empty").mkdir()
    with pytest.raises(ConfigError, match=r"neither a run directory \(metrics.jsonl, config.resolved.yaml, "
                                          r"matches.db or tournaments/\)"):
        write_report(tmp_path / "empty")


def test_a_failing_section_is_a_note_and_the_page_is_still_written(tmp_path, monkeypatch, caplog):
    """Sections are isolated as in the dashboard (ruling C-M4): a failing section becomes a "This section failed"
    note, the other sections and the page are written, the traceback is logged, and ``write_report`` raises
    ``ReportSectionsFailed`` naming the section after writing (the CLI exits 1)."""
    from colosseum.arena.report.data import ReportBuilder
    from colosseum.arena.report.render import ReportSectionsFailed

    run = write_run_dir(tmp_path, "wdl")
    monkeypatch.setattr(ReportBuilder, "_matrix", lambda self, **kw: 1 / 0)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.report.data"):
        with pytest.raises(ReportSectionsFailed, match="matrix") as info:
            write_report(run.root)
    assert info.value.path == run.root / REPORT_FILE
    assert info.value.failed == {"matrix": "ZeroDivisionError: division by zero"}
    data = _data_of(info.value.path.read_text())
    assert set(data["sections"]) == set(SECTIONS)
    assert data["sections"]["matrix"] == {"notes": ["This section failed: ZeroDivisionError: division by zero"]}
    assert data["sections"]["ratings"]["layouts"] and data["sections"]["summary"]["status"] == "finished"
    assert "section matrix failed" in caplog.text and "ZeroDivisionError" in caplog.text


def test_an_unfinished_tournament_dir_is_refused_with_its_own_message(tmp_path):
    tdir = tmp_path / "t1"
    tdir.mkdir()
    (tdir / "tournament.yaml").write_text("mode: select\n")
    with pytest.raises(ConfigError, match="the tournament has no report.json yet"):
        write_report(tdir)


def test_a_tournament_output_dir_counts_as_a_run_dir(tmp_path):
    """``colosseum tournament --output DIR`` lays DIR out like a run dir (ruling C5): ``matches.db`` and
    ``tournaments/`` without ``metrics.jsonl`` or a resolved config; its report is a page."""
    (tmp_path / "out" / "tournaments").mkdir(parents=True)
    assert is_run_dir(tmp_path / "out") and not is_run_dir(tmp_path)
    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "matches.db").write_bytes(b"")
    assert is_run_dir(tmp_path / "db")


def test_a_tournament_output_dir_builds_a_page(tmp_path):
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, RunInfo

    out = tmp_path / "out"
    tdir = out / "tournaments" / "sel"
    tdir.mkdir(parents=True)
    MatchLog(out / MATCH_LOG_FILE, RunInfo(run_id="r1", name="out", parent_run_id=None, game="g",
                                          config_hash="h", checkpoint_interval=10)).close()
    (tdir / "tournament.yaml").write_text("mode: select\n")
    (tdir / "report.json").write_text(json.dumps(sample_report_data()["sections"]["tournaments"]["items"][0]["report"]))
    path = write_report(out)
    assert path == out / REPORT_FILE
    data = _data_of(path.read_text())
    assert set(data["sections"]) == set(SECTIONS)
    assert [item["name"] for item in data["sections"]["tournaments"]["items"]] == ["sel"]
    assert data["sections"]["tournaments"]["items"][0]["report"]["winner"] == "main@v2000"


def test_a_tournament_dir_gets_a_tournament_only_page(tmp_path):
    tdir = tmp_path / "t1"
    tdir.mkdir()
    (tdir / "tournament.yaml").write_text("mode: select\n")
    assert not is_tournament_dir(tdir)
    report = sample_report_data()["sections"]["tournaments"]["items"][0]["report"]
    (tdir / "report.json").write_text(json.dumps(report))
    assert is_tournament_dir(tdir)
    data = _data_of(write_tournament_html(tdir).read_text())
    assert set(data["sections"]) == {"tournaments"}
    assert data["sections"]["tournaments"]["items"][0]["report"]["winner"] == "main@v2000"
    assert _data_of(write_report(tdir).read_text()) == data


def test_write_report_of_a_tournament_dir_uses_the_tournament_writer(tmp_path, monkeypatch):
    """One writer of a tournament dir's ``report.html`` (``write_tournament_html``); ``-o`` writes elsewhere."""
    from colosseum.arena.report import render

    tdir = tmp_path / "t1"
    tdir.mkdir()
    (tdir / "tournament.yaml").write_text("mode: select\n")
    (tdir / "report.json").write_text(json.dumps(sample_report_data()["sections"]["tournaments"]["items"][0]["report"]))
    calls = []
    monkeypatch.setattr(render, "write_tournament_html", lambda d: calls.append(d) or d / REPORT_FILE)
    assert write_report(tdir) == tdir / REPORT_FILE and calls == [tdir]
    other = write_report(tdir, tmp_path / "t.html")
    assert other.exists() and calls == [tdir] and set(_data_of(other.read_text())["sections"]) == {"tournaments"}
