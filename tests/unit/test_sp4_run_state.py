"""run_state.json (SP4 spec block 5, T7.1): written by the hub (SP5; the SP4 launcher) at start, normal end and
failure; readers decide liveness by the pid (and its start time on Linux), never by timestamps."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import pytest

from cli_runner import TTT_CONFIG, training_process, wait_for
from colosseum.core.run_state import (
    RUN_STATE_FILE,
    RunState,
    effective_status,
    process_alive,
    read_run_state,
    write_run_state,
)
from game_helpers import write_test_config


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_round_trip_keeps_keys_it_does_not_own(tmp_path):
    (tmp_path / RUN_STATE_FILE).write_text(json.dumps({"run_id": "abc", "status": "running", "pid": 1}))
    write_run_state(tmp_path, "finished", os.getpid(), exit_code=0)
    state = read_run_state(tmp_path)
    assert (state.status, state.pid, state.exit_code) == ("finished", os.getpid(), 0)
    assert state.extra == {"run_id": "abc"}
    assert json.loads((tmp_path / RUN_STATE_FILE).read_text())["run_id"] == "abc"


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"status": "paused", "pid": 3}), "[]"])
def test_missing_or_malformed_files_read_as_none(tmp_path, content):
    if content is not None:
        (tmp_path / RUN_STATE_FILE).write_text(content)
    assert read_run_state(tmp_path) is None
    assert effective_status(None) == ("unknown", "no run_state.json")


def test_unknown_status_is_refused(tmp_path):
    with pytest.raises(ValueError, match="running"):
        write_run_state(tmp_path, "paused", os.getpid())


def test_liveness_by_pid_and_start_time_not_by_clock(tmp_path):
    write_run_state(tmp_path, "running", os.getpid())
    state = read_run_state(tmp_path)
    assert process_alive(state) and effective_status(state)[0] == "running"
    assert not process_alive(RunState("running", _dead_pid()))
    assert effective_status(RunState("running", _dead_pid()))[0] == "stopped"
    if state.proc_start is not None:        # Linux: a reused pid (other start time) is not the run
        assert not process_alive(RunState("running", os.getpid(), proc_start=state.proc_start + 1))
    assert effective_status(RunState("finished", 5, exit_code=130))[1].startswith("exit code 130")
    assert effective_status(RunState("failed", 5, exit_code=1)) == ("failed", "exit code 1")


def test_training_writes_running_then_finished_after_sigint(tmp_path):
    with training_process(TTT_CONFIG, tmp_path, overrides={"training.total_timesteps": "10000000"}) as (proc, root):
        assert wait_for(lambda: read_run_state(root) is not None, timeout=60)
        state = read_run_state(root)
        assert state.status == "running" and state.pid == proc.pid and process_alive(state)
        assert wait_for(lambda: (root / "metrics.jsonl").exists(), timeout=60)
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=60) == 130
    state = read_run_state(root)
    assert (state.status, state.exit_code) == ("finished", 130)


def test_a_launch_exception_marks_the_run_failed(tmp_path, monkeypatch, restore_root_logging, restore_global_rng):
    import colosseum.hub.hub as hub

    def boom(self):
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(hub.Hub, "launch", boom)
    config = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs"), "name": "r"})
    with pytest.raises(RuntimeError, match="synthetic"):
        hub.run_training(str(config))
    state = read_run_state(tmp_path / "runs" / "r")
    assert (state.status, state.exit_code, state.pid) == ("failed", 1, os.getpid())


def test_a_ctrl_c_during_the_start_marks_the_run_stopped(tmp_path, monkeypatch, restore_root_logging,
                                                          restore_global_rng):
    """A KeyboardInterrupt inside ``launch`` before its signal handlers exist (the CLI exits 130): ``stopped`` /
    130, not ``failed`` / 1."""
    import colosseum.hub.hub as hub

    def interrupted(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(hub.Hub, "launch", interrupted)
    config = write_test_config(tmp_path / "cfg.yaml", "turns", run={"dir": str(tmp_path / "runs"), "name": "r"})
    with pytest.raises(KeyboardInterrupt):
        hub.run_training(str(config))
    state = read_run_state(tmp_path / "runs" / "r")
    assert (state.status, state.exit_code) == ("stopped", 130)
    status, detail = effective_status(state)
    assert status == "stopped" and detail.startswith("exit code 130") and "start" in detail
