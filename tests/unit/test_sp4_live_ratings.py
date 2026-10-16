"""Live ratings (SP4 T2.2, spec blocks 2 and 7.3): the refit thread publishes snapshots and skips an unchanged
table, survives a failing refit, logs anchor notes once; the bt section of ratings.json, the compact ratings record,
the console and WandB fields; the ratings config section and its validate checks; one BLAS thread in the training
main process."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from click.testing import CliRunner

from colosseum.arena.comparisons import ComparisonTable
from colosseum.arena.live import (
    LiveRatings,
    bt_section,
    bt_summary,
    compute_snapshot,
    console_bt,
    random_bot_agent,
    rating_inputs,
)
from colosseum.core.config import ColosseumConfig, RatingsConfig
from colosseum.core.errors import ConfigError
from colosseum.core.registry import validate_config
from colosseum.league.pfsp import compact_ratings, pfsp_summary
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import MetricsWriter
from game_helpers import (
    bt_table,
    make_coordinator,
    make_match_result,
    make_test_config,
    scripted_agent,
    synthetic_bt_matches,
)


def _wait(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _league_table(n: int = 600, seed: int = 0) -> ComparisonTable:
    strengths = {"random": -1.0, "greedy": 0.3, "main@v0": -0.5, "main@v100": 0.2, "main@v200": 0.6}
    return bt_table(synthetic_bt_matches(strengths, n_matches=n, rng=np.random.default_rng(seed), bias=[0.2, -0.2]),
                    interval=100)


def _snapshot(table: ComparisonTable):
    return compute_snapshot(table, version=1, env_steps=1234, fixed_agents=("random", "greedy"), random_bot="random",
                            agent_order=("main", "random", "greedy"), top_k=2, rng=np.random.default_rng(0),
                            samples=300)


def test_the_thread_publishes_snapshots_and_skips_an_unchanged_table():
    table = _league_table()
    steps = [500]
    live = LiveRatings(table, interval_sec=0.05, fixed_agents=("random", "greedy"), random_bot="random",
                       env_steps_fn=lambda: steps[0], samples=200)
    assert live.latest() is None and live.last_refit_sec is None
    live.start()
    try:
        assert _wait(lambda: live.latest() is not None)
        first = live.latest()
        assert first.version == 1 and first.env_steps == 500 and set(first.fits) == {"2p"}
        time.sleep(0.3)
        assert live.latest() is first                      # unchanged table: no new refit
        steps[0] = 900
        table.add_result(make_match_result([(0, "main", "ckpt_v200"), (1, "random", "fixed")]), [200, None])
        assert _wait(lambda: live.latest().version == 2)
        assert live.latest().env_steps == 900 and live.last_refit_sec >= 0.0
    finally:
        live.stop(timeout=5.0)


def test_a_failing_refit_is_logged_once_and_the_thread_goes_on(caplog, monkeypatch):
    table = _league_table()
    calls = []
    original = table.copy

    def flaky_copy():
        calls.append(1)
        if len(calls) <= 3:
            raise RuntimeError("table broken")
        return original()

    monkeypatch.setattr(table, "copy", flaky_copy)
    live = LiveRatings(table, interval_sec=0.02, env_steps_fn=lambda: 0, samples=100)
    with caplog.at_level(logging.ERROR, logger="colosseum.arena.live"):
        live.start()
        try:
            assert _wait(lambda: live.latest() is not None)
        finally:
            live.stop(timeout=5.0)
    errors = [r for r in caplog.records if "refit failed" in r.getMessage()]
    assert len(errors) == 3 and errors[0].exc_info and not errors[1].exc_info   # a traceback the first time only


def test_anchor_notes_are_logged_once(caplog):
    live = LiveRatings(_league_table(), interval_sec=3600.0, random_bot="nobot", env_steps_fn=lambda: 0, samples=50)
    with caplog.at_level(logging.INFO, logger="colosseum.arena.live"):
        live.refit_now()
        live._table.add_result(make_match_result([(0, "main", "ckpt_v0"), (1, "random", "fixed")]), [0, None])
        live.refit_now()
    notes = [r.getMessage() for r in caplog.records if "did not play here" in r.getMessage()]
    assert len(notes) == 1 and "RandomBot agent 'nobot'" in notes[0]


def test_the_bt_section_of_ratings_json():
    table = _league_table()
    table.add_result(make_match_result([(0, "a", "fixed"), (0, "b", "fixed")], scores=[3.0], layout="coop"),
                     [None, None])
    section = bt_section(_snapshot(table))
    assert set(section) == {"2p", "coop"}
    rows = section["2p"]["player"]
    assert [r["rating"] for r in rows] == sorted((r["rating"] for r in rows), reverse=True)
    assert {r["entity"] for r in rows} == {"random", "greedy", "main@v0", "main@v100", "main@v200"}
    (anchor,) = [r for r in rows if r["anchor"]]
    assert anchor["entity"] == "random" and anchor["rating"] == 0.0
    main = next(r for r in rows if r["entity"] == "main@v200")
    assert set(main) >= {"lineage", "version", "se", "ci_low", "ci_high", "lcb", "games", "p_best", "p_best_lineage",
                         "p_top_k", "component"}
    assert main["lineage"] == "main" and main["version"] == 200 and main["ci_low"] < main["rating"] < main["ci_high"]
    assert set(section["2p"]["@bias"]) == {"0", "1"} and section["2p"]["@anchors"]["player/0"]["rule"] == "random_bot"
    assert "p_value" in section["2p"]["@transitivity"] and section["2p"]["@fit"]["converged"]
    assert section["coop"]["@score"]["compositions"][0]["composition"] == "a+b"
    json.dumps(section)                                    # JSON-ready
    assert bt_section(None) == {}


def test_the_compact_bt_summary_and_the_console_field():
    summary = bt_summary(_snapshot(_league_table()), ["main", "absent"])
    assert set(summary) == {"main"}
    cells = summary["main"]["2p"]
    assert cells["latest"]["entity"] == "main@v200"
    assert cells["best"]["rating"] == max(cells["best"]["rating"], cells["latest"]["rating"])
    rating, half = console_bt(summary["main"])
    assert rating == cells["latest"]["rating"] and half == pytest.approx(
        (cells["latest"]["ci_high"] - cells["latest"]["ci_low"]) / 2)
    assert console_bt({}) is None and bt_summary(None, ["main"]) == {}


def test_the_console_line_rounds_the_bt_field():
    from colosseum.metrics.console import ConsoleReporter

    def line(bt):
        return ConsoleReporter(100).format_line("main", train_step=1, env_steps=10, fps=1.0, loss=None, entropy=None,
                                                return_mean=None, wr_vs_past=None, wr_arena=None, bt=bt)

    assert line((-0.3, 61.6)).endswith("| bt 0 ± 62") and line((123.6, 7.2)).endswith("| bt 124 ± 7")
    assert "bt" not in line(None)


def test_pfsp_tables_are_summarized_in_the_records():
    table = {"a": {"a@ckpt_v1": {"score": 0.25, "games": 4.0}, "rnd@fixed": {"score": 0.75, "games": 2.0}},
             "b": {}}
    assert pfsp_summary(table) == {"a": {"players": 2, "min_score": 0.25, "max_score": 0.75},
                                   "b": {"players": 0, "min_score": None, "max_score": None}}
    compact = compact_ratings({"2p": {"elo": {"a": 1200.0}, "pfsp": table}})
    assert compact == {"2p": {"elo": {"a": 1200.0}, "pfsp_summary": pfsp_summary(table)}}


def test_the_coordinator_gives_the_compact_snapshot(tmp_path):
    coordinator = make_coordinator(make_test_config("turns"), tmp_path / "ckpt")
    full, compact = coordinator.ratings_snapshot(), coordinator.ratings_snapshot(compact=True)
    assert "pfsp" in full["2p"] and "pfsp" not in compact["2p"] and "pfsp_summary" in compact["2p"]


def test_the_hub_writes_compact_records_bt_in_ratings_json_console_and_wandb(tmp_path, caplog):
    rows = []

    class FakeWandB:
        def log_global(self, row, step):
            rows.append(row)

        def log_train(self, agent_id, values, step):
            pass

    clock = [0.0]
    built = []
    full = {"2p": {"elo": {"main": 1200.0}, "win_rates": {}, "games": {}, "wr_vs_past": {}, "past_games": {},
                   "scores": {}, "cross_play": {}, "role_win_rates": {},
                   "pfsp": {"main": {"random@fixed": {"score": 0.4, "games": 3.0}}}}}

    def ratings():
        built.append(1)
        return full

    hub = MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                     agent_ids=["main"], total_timesteps=100, log_interval=1, console_interval_sec=10.0,
                     wandb_logger=FakeWandB(), clock=lambda: clock[0])
    snapshot = _snapshot(_league_table())
    for _ in range(5):
        hub.maybe_tick(env_steps=10, ratings=ratings, queue_depths={}, bt=snapshot)    # not due yet
    assert built == []                                     # built only at a tick (spec block 7.3)
    clock[0] = 11.0
    with caplog.at_level(logging.INFO, logger="colosseum.progress"):
        hub.close(env_steps=20, ratings=ratings, queue_depths={}, bt=snapshot)
    assert built == [1]
    record = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
              if '"ratings"' in line][-1]
    assert "pfsp" not in record["layouts"]["2p"] and record["layouts"]["2p"]["pfsp_summary"]["main"]["players"] == 1
    assert record["bt"]["main"]["2p"]["latest"]["entity"] == "main@v200"
    written = json.loads((tmp_path / "ratings.json").read_text())
    assert set(written) == {"env_steps", "layouts", "bt"} and "pfsp" in written["layouts"]["2p"]
    assert written["bt"]["2p"]["player"][0]["rating"] >= written["bt"]["2p"]["player"][-1]["rating"]
    assert "| bt " in caplog.text and " ± " in caplog.text
    assert any(key.startswith("ratings/bt/main/2p/latest/rating") for key in rows[-1])
    assert not any("/pfsp" in key for key in rows[-1])


def test_without_a_snapshot_the_record_has_no_bt_and_ratings_json_an_empty_one(tmp_path):
    hub = MetricsHub(writer=MetricsWriter(tmp_path / "m.jsonl"), ratings_path=tmp_path / "r.json", agent_ids=["a"],
                     total_timesteps=10, log_interval=1, console_interval_sec=0.0)
    hub.close(env_steps=0, ratings={}, queue_depths={})
    record = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines() if '"ratings"' in line][0]
    assert "bt" not in record and json.loads((tmp_path / "r.json").read_text())["bt"] == {}


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def test_the_ratings_section_its_validation_and_the_random_bot_rule():
    cfg = ColosseumConfig.model_validate(_data())
    assert cfg.ratings == RatingsConfig(anchor=None, refit_interval_sec=60.0, bootstrap_samples=200)
    for bad in ({"refit_interval_sec": 0}, {"bootstrap_samples": 10}, {"anchors": "x"}):
        with pytest.raises(ValueError):
            ColosseumConfig.model_validate(_data(ratings=bad))
    league = make_test_config("turns", agents={"main": {}, "greedy": scripted_agent("game_helpers.ConstantBot"),
                                               "rnd": scripted_agent()})
    assert random_bot_agent(league) == "rnd"
    options, fixed, random_bot = rating_inputs(league)
    assert (options.anchor, options.agent_order, fixed, random_bot) == (None, ("main", "greedy", "rnd"),
                                                                         ["greedy", "rnd"], "rnd")
    report = validate_config(league)
    assert any("anchor 'rnd' (the first RandomBot agent)" in line for line in report.lines)
    assert any(line.startswith("match log: on") for line in report.lines)
    with pytest.raises(ConfigError, match="ratings.anchor 'nobody' is not an agent"):
        validate_config(make_test_config("turns", ratings={"anchor": "nobody"}))


def test_train_limits_the_main_process_blas_before_numpy(monkeypatch):
    import colosseum.launcher
    import colosseum.utils.process as process

    calls = []
    monkeypatch.setattr(process, "limit_main_blas_threads", lambda: calls.append("blas") or False)
    monkeypatch.setattr(colosseum.launcher, "run_training", lambda *a, **k: calls.append("train") or 0)
    from colosseum.cli import main

    result = CliRunner().invoke(main, ["train", "-c", os.devnull])
    assert result.exit_code == 0 and calls == ["blas", "train"]


_BLAS_PROBE = r"""
import ctypes, glob, multiprocessing as mp, os, sys
from colosseum.utils.process import limit_main_blas_threads

def threads():
    import numpy
    libs = glob.glob(os.path.join(os.path.dirname(numpy.__file__), "..", "numpy.libs", "*openblas*.so*"))
    if not libs:
        return None
    lib = ctypes.CDLL(libs[0])
    for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
        if hasattr(lib, name):
            return getattr(lib, name)()
    return None

def child(q):
    q.put(threads())

if __name__ == "__main__":
    before = dict(os.environ)
    assert limit_main_blas_threads() is True
    assert dict(os.environ) == before, "the environment must be restored"
    assert limit_main_blas_threads() is False          # numpy is imported now
    mp.set_start_method("spawn")
    q = mp.Queue()
    p = mp.Process(target=child, args=(q,))
    p.start()
    print(threads(), q.get())
    p.join()
"""


def test_the_blas_limit_applies_to_this_process_only(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(_BLAS_PROBE)
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                           "MKL_NUM_THREADS")}
    out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    main_threads, child_threads = out.stdout.split()
    if main_threads == "None":
        pytest.skip("numpy does not use a bundled OpenBLAS here")
    assert int(main_threads) == 1
    if (os.cpu_count() or 1) > 1:
        assert int(child_threads) > 1                     # spawned children keep the default
