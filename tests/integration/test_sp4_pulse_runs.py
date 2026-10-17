"""Pulse in real runs (SP4 T6.1): snapshots evaluated into the journal on the same seeds; a pulse crash stops the
run with a pointer to its log; the pulse dies with the run. One rollout worker + pulse = two worker processes."""
from __future__ import annotations

import json
import os
import re
import signal

from cli_runner import read_records, run_train, training_process, wait_for
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog
from game_helpers import scripted_agent, write_test_config

AGENTS = {"main": {}, "rnd": scripted_agent()}
PULSE = {"evaluation.enabled": "true", "evaluation.every": "1", "evaluation.blocks": "2", "evaluation.num_envs": "2",
         "evaluation.max_backlog": "2", "training.total_timesteps": "4000", "checkpoint.interval": "10"}


def test_pulse_evaluates_snapshots_into_the_journal_on_the_same_seeds(tmp_path):
    cfg = write_test_config(tmp_path / "turns.yaml", "turns", agents=AGENTS)
    run = run_train(cfg, tmp_path, name="pulse", overrides=PULSE)
    assert run.returncode == 0, run.stderr[-3000:]
    state = json.loads((run.root / "pulse_state.json").read_text())
    offered, evaluated = state["offered"]["main"], state["evaluated"]["main"]
    rest = set(state["skipped"]["main"]) | set(state["unevaluated"]["main"])
    assert evaluated and set(offered) == set(evaluated) | rest
    reader = MatchLog.open_readonly(run.root / MATCH_LOG_FILE)
    try:
        results = reader.match_results(source="pulse")
    finally:
        reader.close()
    by_version: dict[int, set] = {}
    for r in results:
        version = next(s.version_start for s in r.seats if s.agent_id == "main")
        by_version.setdefault(version, set()).add(r.seed)
        meta = run.root / "checkpoints" / "main" / f"ckpt_v{version}" / "meta.json"
        if meta.is_file():
            assert r.env_steps == json.loads(meta.read_text())["env_steps"]
    assert set(by_version) == set(evaluated)
    assert len({frozenset(seeds) for seeds in by_version.values()}) == 1    # identical seeds for every snapshot
    assert (run.root / "logs" / "pulse.log").is_file()
    system = read_records(run.root, "system")
    assert system and "pulse_skipped_snapshots" in system[-1]
    main_log = run.log("main")
    assert "Terminated processes" not in main_log                          # the pulse exits by itself at the stop
    m = re.search(r"Match results: (\d+) received \((\d+) from workers, (\d+) from the pulse\), (\d+) journaled, "
                  r"(\d+) duplicates, (\d+) dropped by the journal writer", main_log)
    total, workers, pulse, written, duplicates, dropped = map(int, m.groups())
    assert total == workers + pulse == written + duplicates + dropped and pulse == len(results)


def test_a_pulse_crash_stops_the_run_with_a_pointer_to_its_log(tmp_path):
    cfg = write_test_config(tmp_path / "turns.yaml", "turns",
                            agents={"main": {}, "bad": scripted_agent("game_helpers.PulseFailBot")})
    run = run_train(cfg, tmp_path, name="crash", overrides={**PULSE, "training.total_timesteps": "200000"})
    assert run.returncode == 1
    log = run.log("main")
    assert "pulse died" in log and "pulse.log" in log
    assert "pulse failure on purpose" in run.log("pulse")


def test_the_pulse_dies_with_the_run(tmp_path):
    cfg = write_test_config(tmp_path / "turns.yaml", "turns", agents=AGENTS)
    with training_process(cfg, tmp_path, name="orphan", overrides={**PULSE, "training.total_timesteps": "10000000"}) \
            as (proc, root):
        log = root / "logs" / "pulse.log"
        assert wait_for(lambda: log.is_file() and "started (pid" in log.read_text(), 120)
        pid = int(re.search(r"started \(pid (\d+)\)", log.read_text()).group(1))
        os.kill(proc.pid, signal.SIGKILL)                                   # the main process only

        def gone() -> bool:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            return False

        assert wait_for(gone, 10)
