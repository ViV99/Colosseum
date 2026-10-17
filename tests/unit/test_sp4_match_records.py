"""Match records of SP4 (T1.1, spec block 1): the new optional fields, rating entities, per-seat weights versions
read by MatchRunner at the episode's start and end, explicit lineup seeds / blocks / uids, the uid prefix, the
run id file and the match_log config section."""
from __future__ import annotations

import json

import numpy as np
import pytest

from colosseum.arena.entities import entity_key, entity_version, lineage, rating_version, snapshot_version
from colosseum.core.config import ColosseumConfig, MatchLogConfig, load_config
from colosseum.core.errors import ConfigError
from colosseum.core.run_dir import RUN_INFO_FILE, RunDir, read_run_id
from colosseum.core.types import MATCH_SOURCES, Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.envs.vector import VectorEnv
from colosseum.eval import play_lineups
from colosseum.worker.match_runner import MatchRunner
from game_helpers import DictModelPool, RecordingObserver, Tick, TickGame, make_test_config, make_test_model

TURNS = [Tick(acting={0}), Tick(acting={1}), Tick(acting={0}), Tick(over=True, rewards={0: 1.0, 1: -1.0})]
ROLE2 = TickGame(TURNS, 2).spec.roles["player"]


def test_new_record_fields_have_defaults():
    seat = SeatResult(0, "player", 0, "a", "latest", 1.0)
    assert (seat.version_start, seat.version_end) == (None, None)
    result = MatchResult("m", "2p", "wdl", [seat], [TeamResult(0, 1.0, 1.0)], 3)
    assert (result.uid, result.source, result.seed, result.env_steps, result.block) == ("", "train", None, None, None)
    lineup = Lineup("2p", [SeatAssignment("a")])
    assert (lineup.seed, lineup.block, lineup.uid) == (None, None, None)
    assert MATCH_SOURCES == ("train", "tournament", "pulse")


def test_entities_versions_buckets_and_lineages():
    assert snapshot_version("ckpt_v1200") == 1200
    assert snapshot_version("latest") is None and snapshot_version("fixed") is None
    assert rating_version("ckpt_v1250", 99, 100) == 1250                 # a snapshot counts for its own version
    assert rating_version("latest", 1250, 100) == 1200                   # latest counts for its bucket
    assert rating_version("latest", 99, 100) == 0
    assert rating_version("latest", None, 100) is None
    assert rating_version("fixed", 7, 100) is None
    with pytest.raises(ValueError, match="network id"):
        rating_version("ckpt_x", 1, 100)
    assert entity_key("main", 1200) == "main@v1200" and entity_key("random", None) == "random"
    assert lineage("main@v1200") == "main" and lineage("b/main@v3") == "b/main" and lineage("greedy") == "greedy"
    assert entity_version("main@v1200") == 1200 and entity_version("prev_sub#ab12cd34") is None


def _runner(lineups, *, versions=None, seed=0, num_envs=1, **kwargs):
    games: list[TickGame] = []

    def env_fn():
        games.append(TickGame(TURNS, 2))
        return games[-1]

    model = make_test_model(ROLE2)
    pool = DictModelPool({("a", "latest"): model, ("b", "latest"): model, ("b", "ckpt_v7"): model,
                          ("bot", "fixed"): model})
    observer = RecordingObserver()
    runner = MatchRunner(vec_env=VectorEnv(env_fn, num_envs), lineups=lineups, models=pool, observer=observer,
                         seed=seed, version_fn=None if versions is None else versions.get, **kwargs)
    return runner, observer, games


def _results(observer):
    return [e[2].result for e in observer.events if e[0] == "end"]


def test_seat_versions_at_the_episode_start_and_end_follow_the_version_function():
    versions = {"a": 3}
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b", "ckpt_v7", collect=False)])
    runner, observer, _ = _runner([lineup], versions=versions)
    runner.step()
    versions["a"] = 5                             # new weights arrive in the middle of the episode
    runner.step()
    runner.step()                                 # episode 0 ends (3 steps)
    for _ in range(3):
        runner.step()                             # episode 1 starts and ends with version 5
    first, second = _results(observer)
    assert [(s.version_start, s.version_end) for s in first.seats] == [(3, 5), (7, 7)]
    assert [(s.version_start, s.version_end) for s in second.seats] == [(5, 5), (7, 7)]


def test_fixed_seats_and_runners_without_a_version_source_have_no_versions():
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("bot", "fixed", collect=False)])
    runner, observer, _ = _runner([lineup])
    for _ in range(3):
        runner.step()
    (result,) = _results(observer)
    assert [(s.version_start, s.version_end) for s in result.seats] == [(None, None), (None, None)]


def test_derived_seeds_uids_and_sources_of_runner_made_results():
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")])
    runner, observer, games = _runner([lineup], seed=11, uid_prefix="RUN/train/", match_id_prefix="w2_e",
                                      source="pulse")
    for _ in range(3):
        runner.step()
    (result,) = _results(observer)
    expected = int(np.random.SeedSequence([11, 0, 0]).generate_state(1)[0])
    assert games[0].log[0][2] == expected and result.seed == expected
    assert result.match_id == "w2_e0_ep0" and result.uid == "RUN/train/w2_e0_ep0"
    assert result.source == "pulse" and result.block is None and result.env_steps is None
    runner_without_seed, observer2, _ = _runner([lineup], seed=None)
    for _ in range(3):
        runner_without_seed.step()
    assert _results(observer2)[0].seed is None and _results(observer2)[0].uid == "m0_ep0"


def test_explicit_lineup_seeds_blocks_and_uids_reach_the_env_and_the_result():
    first = Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")], seed=1234, block="A/0/7", uid="T1/A/0/7/r0")
    second = Lineup("2p", [SeatAssignment("b"), SeatAssignment("a")], seed=1234, block="A/0/7", uid="T1/A/0/7/r1")
    runner, observer, games = _runner([first], seed=5)
    runner.set_next_lineup(0, second)
    for _ in range(6):
        runner.step()
    one, two = _results(observer)[:2]
    assert [log[2] for log in games[0].log if log[3] is None][:2] == [1234, 1234]   # both resets
    assert (one.seed, one.block, one.uid) == (1234, "A/0/7", "T1/A/0/7/r0")
    assert (two.seed, two.block, two.uid) == (1234, "A/0/7", "T1/A/0/7/r1")
    assert [s.agent_id for s in two.seats] == ["b", "a"]


def test_a_missing_snapshot_keeps_the_lineup_fields():
    lineup = Lineup("2p", [SeatAssignment("a"), SeatAssignment("a", "ckpt_v99", collect=False)], seed=9,
                    block="B", uid="U")
    runner, observer, _ = _runner([lineup])
    assert (runner.lineup(0).seed, runner.lineup(0).block, runner.lineup(0).uid) == (9, "B", "U")
    assert runner.lineup(0).seats[1].network_id == "latest"          # SP1 fallback


def test_an_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="source"):
        _runner([Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")])], source="eval")


def test_play_lineups_keeps_explicit_seeds():
    model = make_test_model(ROLE2)
    games: list[TickGame] = []

    def env_fn():
        games.append(TickGame(TURNS, 2))
        return games[-1]

    results = play_lineups(env_fn=env_fn, models={"a": model, "b": model}, num_envs=1, seed=0,
                           lineups=[Lineup("2p", [SeatAssignment("a"), SeatAssignment("b")], seed=77, uid="x")])
    assert results[0].seed == 77 and results[0].uid == "x" and games[0].log[0][2] == 77


def test_run_id_is_written_once_and_read_without_writing(tmp_path):
    config = make_test_config("turns")
    config.run.dir, config.run.name = str(tmp_path / "runs"), "r"
    run = RunDir.create(config)
    assert run.run_id is None and not (run.root / RUN_INFO_FILE).exists()
    run_id = run.ensure_run_id()
    assert len(run_id) == 32 and run.ensure_run_id() == run_id == read_run_id(run.root)
    assert json.loads((run.root / RUN_INFO_FILE).read_text()) == {"run_id": run_id}
    assert read_run_id(tmp_path / "nowhere") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / RUN_INFO_FILE).write_text("{")
    with pytest.raises(ConfigError, match="run.json"):
        read_run_id(tmp_path / "bad")


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def test_match_log_config_defaults_validation_and_overrides(tmp_path):
    cfg = ColosseumConfig.model_validate(_data())
    assert cfg.match_log == MatchLogConfig(enabled=True, detail_window_mb=512.0, min_free_disk_mb=2048.0,
                                           compact_interval_sec=60.0)
    for bad in ({"detail_window_mb": 0}, {"min_free_disk_mb": -1}, {"compact_interval_sec": 0}, {"window": 1}):
        with pytest.raises(ValueError):
            ColosseumConfig.model_validate(_data(match_log=bad))
    path = tmp_path / "cfg.yaml"
    path.write_text(json.dumps(_data()))
    assert load_config(path, {"match_log.detail_window_mb": 64}).match_log.detail_window_mb == 64.0
    with pytest.raises(ConfigError):                       # global only: no per-agent override
        load_config(path, {"agents.a.match_log.enabled": False})
    assert ColosseumConfig.model_validate(_data(agents={"a": {}})).get_agent_config("a").match_log.enabled
