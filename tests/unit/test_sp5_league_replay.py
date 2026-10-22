"""The league replay of an in-place resume (SP5 T4.3, R8, spec 5.1 / 5.4): ``league_state.json``'s journal
watermark, the replay window for SP5 / SP4 / missing league states, and exactly-once accounting: a coordinator
that loads the state and replays the journal rows after the watermark equals one that accounted every result."""
from __future__ import annotations

import json

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog, MatchLogReader, RunInfo
from colosseum.coordinator.coordinator import LEAGUE_STATE_FILE, LeagueStateInfo, league_replay_window
from game_helpers import make_coordinator, make_match_result, make_test_config, scripted_agent

RUN = RunInfo(run_id="R", name="run", parent_run_id=None, game="turns", config_hash="h", checkpoint_interval=10)


def _config():
    return make_test_config("turns", agents={"agent_0": {}, "rnd": scripted_agent()})


def _results(n: int = 100):
    out = []
    for k in range(n):
        other = ("rnd", "fixed") if k % 3 else ("agent_0", "ckpt_v10")
        ranks = [1.0, 2.0] if k % 4 else ([2.0, 1.0] if k % 8 else [1.0, 1.0])
        out.append(make_match_result([(0, "agent_0", "latest", "player", k), (1, *other)], ranks,
                                     uid=f"R/train/n1/w0_e0_g1_ep{k}", env_steps=10 * (k + 1)))
    return out


def _journal(tmp_path, results):
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    for k in range(0, len(results), 7):
        log.append(results[k:k + 7])
    log.close()
    return tmp_path / MATCH_LOG_FILE


def _replayed(tmp_path, config, state_path):
    fresh = make_coordinator(config, tmp_path / "fresh")
    if state_path is not None:
        assert fresh.load_league_state(state_path)
    after, min_steps = league_replay_window(fresh.loaded_league_state)
    with MatchLogReader(tmp_path / MATCH_LOG_FILE) as reader:
        replayed, skipped = fresh.replay_results(res for _, res in reader.results_after(after,
                                                                                        min_env_steps=min_steps))
    return fresh, replayed, skipped


def test_the_watermark_is_saved_and_loaded(tmp_path):
    coordinator = make_coordinator(_config(), tmp_path / "ckpt")
    path = tmp_path / LEAGUE_STATE_FILE
    coordinator.save_league_state(path, env_steps=5, journal_row=42)
    assert json.loads(path.read_text())["journal_row"] == 42
    fresh = make_coordinator(_config(), tmp_path / "ckpt2")
    assert fresh.loaded_league_state is None and fresh.league_state_journal_row() is None
    fresh.load_league_state(path)
    assert fresh.loaded_league_state == LeagueStateInfo(env_steps=5, journal_row=42, has_watermark=True)
    assert fresh.league_state_journal_row() == 42


def test_the_replay_window():
    assert league_replay_window(None) == (None, None)                                   # no state: every row
    assert league_replay_window(LeagueStateInfo(7, 42, True)) == (42, None)             # SP5: after the watermark
    assert league_replay_window(LeagueStateInfo(7, None, True)) == (None, None)         # SP5, nothing accounted
    assert league_replay_window(LeagueStateInfo(700, None, False)) == (None, 700)       # SP4: after its env steps


def test_replay_after_the_watermark_is_exactly_once(tmp_path):
    config, results = _config(), _results()
    _journal(tmp_path, results)
    full = make_coordinator(config, tmp_path / "full")
    for result in results[:60]:
        full.report_match_result(result)
    state = tmp_path / LEAGUE_STATE_FILE
    full.save_league_state(state, env_steps=600, journal_row=60)      # rows 1..60 accounted
    for result in results[60:]:                                        # committed, accounted, then the hub died
        full.report_match_result(result)
    fresh, replayed, skipped = _replayed(tmp_path, config, state)
    assert (replayed, skipped) == (40, 0)
    assert fresh.ratings_snapshot() == full.ratings_snapshot()


def test_an_sp4_league_state_replays_by_env_steps(tmp_path):
    config, results = _config(), _results()
    _journal(tmp_path, results)
    full = make_coordinator(config, tmp_path / "full")
    for result in results[:60]:
        full.report_match_result(result)
    state = tmp_path / LEAGUE_STATE_FILE
    full.save_league_state(state, env_steps=results[59].env_steps)
    data = json.loads(state.read_text())
    del data["journal_row"]                                            # as SP4 wrote it
    state.write_text(json.dumps(data))
    for result in results[60:]:
        full.report_match_result(result)
    fresh, replayed, _ = _replayed(tmp_path, config, state)
    assert replayed == 40 and fresh.ratings_snapshot() == full.ratings_snapshot()


def test_without_a_league_state_every_detailed_match_is_replayed(tmp_path):
    config, results = _config(), _results()
    _journal(tmp_path, results)
    full = make_coordinator(config, tmp_path / "full")
    for result in results:
        full.report_match_result(result)
    fresh, replayed, _ = _replayed(tmp_path, config, None)
    assert replayed == 100 and fresh.ratings_snapshot() == full.ratings_snapshot()


def test_folded_rows_after_the_watermark_are_counted(tmp_path):
    results = _results()
    log = MatchLog(tmp_path / MATCH_LOG_FILE, RUN, disk_free_fn=lambda _: 10**12)
    log.append(results)
    log.compact(window_bytes=0, max_batch=10)                           # every training match folded
    log.close()
    with MatchLogReader(tmp_path / MATCH_LOG_FILE) as reader:
        assert list(reader.results_after(60)) == []
        assert reader.folded_after(60) == 40 and reader.folded_after(None) == 100


def test_a_result_the_game_cannot_rate_is_skipped(tmp_path, caplog):
    config = _config()
    fresh = make_coordinator(config, tmp_path / "c")
    bad = make_match_result([(0, "agent_0", "latest"), (1, "rnd", "fixed")], layout="gone", uid="x")
    with caplog.at_level("WARNING"):
        assert fresh.replay_results([bad, _results(1)[0]]) == (1, 1)
    assert "cannot be replayed" in caplog.text
