"""``colosseum train`` = a hub and its local node (SP5 T2.5; spec 1, 4.3, 4.4, 7.5, 7.7; rulings R5, R12, R18):
the run reaches its budget with the local node's workers and learner, ``hub.json`` / ``hub_state.json`` / the
running registry, node-scoped uids, ``train --listen``, signals with final snapshots, a node's contract error and a
dead local node. Linux (/proc) for the process checks."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cli_runner import (
    REPO_ROOT,
    TTT_CONFIG,
    child_env,
    run_in_session,
    run_train,
    train_cmd,
    training_process,
    wait_for,
)
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.run_dir import read_run_id
from colosseum.hub.registry import read_hub_json
from colosseum.net.security import hub_identity_dir
from game_helpers import write_test_config

pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")
FOREVER = {"training.total_timesteps": "100000000"}


def _descendants(pid: int) -> set[int]:
    table = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text()
            except OSError:
                continue
            table[int(d.name)] = int(stat[stat.rfind(")") + 2:].split()[1])
    found, frontier = set(), [pid]
    while frontier:
        parent = frontier.pop()
        for child, ppid in table.items():
            if ppid == parent and child not in found:
                found.add(child)
                frontier.append(child)
    return found


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z"


def _metas(root: Path, agent: str = "agent_0") -> list[dict]:
    return [json.loads((c / "meta.json").read_text()) for c in sorted((root / "checkpoints" / agent).glob("ckpt_v*"))]


def test_train_runs_a_hub_and_a_local_node_to_the_budget(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="budget")
    assert run.returncode == 0, run.stderr[-3000:]
    logs = {p.name for p in (run.root / "logs").iterdir()}
    assert {"main.log", "node.log", "worker-0.log", "learner-agent_0.log"} <= logs
    info = read_hub_json(run.root)
    assert info.mode == "train" and info.tls is False and info.addresses == ()
    assert re.fullmatch(r"127\.0\.0\.1:[1-9]\d*", info.loopback)
    state = json.loads((run.root / "hub_state.json").read_text())
    assert state["env_steps"] >= 3000 and len(state["sessions"]) == 1 and state["epoch_reserved_until"] == 1024
    run_id = read_run_id(run.root)
    with MatchLogReader(run.root / MATCH_LOG_FILE) as reader:
        uids = [r.uid for r in reader.matches(source="train", limit=None)]
    assert uids and len(uids) == len(set(uids))
    assert all(re.fullmatch(rf"{run_id}/train/n0/w0_e[0-7]_g\d+_ep\d+", uid) for uid in uids), uids[:3]
    metas = _metas(run.root)
    assert metas and all("env_steps" in m for m in metas) and any(m["final"] for m in metas)
    assert not list((hub_identity_dir() / "running").glob("*.json"))      # the running registry entry is gone
    run_state = json.loads((run.root / "run_state.json").read_text())
    assert (run_state["status"], run_state["exit_code"]) == ("finished", 0)
    assert "[ERROR]" not in run.log("main") and "Training budget reached" in run.log("main")


def test_train_listen_opens_a_tls_port_and_keeps_the_loopback(tmp_path):
    cmd = train_cmd(TTT_CONFIG, tmp_path / "runs", "listen", {"distributed.port": "0"}) + ["--listen", "127.0.0.1"]
    proc = run_in_session(cmd, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    root = tmp_path / "runs" / "listen"
    info = read_hub_json(root)
    (address,) = info.addresses
    assert re.fullmatch(r"127\.0\.0\.1:[1-9]\d*", address) and info.tls is True
    assert info.loopback != address and info.loopback.startswith("127.0.0.1:") and not info.loopback_tls
    assert f"session with the hub at {info.loopback} (plain)" in (root / "logs" / "node.log").read_text()


@pytest.mark.parametrize("signum, code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_sigint_and_sigterm_stop_train_with_final_checkpoints(tmp_path, signum, code):
    with training_process(TTT_CONFIG, tmp_path, name="sig", overrides=FOREVER) as (proc, root):
        assert wait_for(lambda: bool(_metas(root)), 120), (tmp_path / "sig.stderr").read_text()[-3000:]
        kids = _descendants(proc.pid)
        assert len(kids) >= 3                                         # the node, its worker and its learner
        t0 = time.monotonic()
        if signum == signal.SIGINT:
            os.killpg(proc.pid, signal.SIGINT)                        # Ctrl-C reaches the whole group
        else:
            proc.send_signal(signal.SIGTERM)
        assert proc.wait(15) == code
        assert wait_for(lambda: not any(_alive(k) for k in kids), max(0.0, 10 - (time.monotonic() - t0)))
    assert any(m["final"] for m in _metas(root)), "no final snapshot after the signal"
    assert "KeyboardInterrupt" not in (tmp_path / "sig.stderr").read_text()


def test_a_contract_error_on_the_node_stops_train_with_exit_1_and_its_context(tmp_path):
    config = write_test_config(tmp_path / "broken.yaml", "turns",
                               env={"env_class": "game_helpers.WorkerContractBreakingGame"},
                               training={"total_timesteps": 100_000_000})
    run = run_train(config, tmp_path, name="contract", tiny=False)
    assert run.returncode == 1, run.stderr[-3000:]
    errors = [line for line in run.log("main").splitlines() if "[ERROR]" in line]
    assert len(errors) == 1, errors
    (error,) = errors
    for part in ("node 0 (", "worker 0", "env", "episode step", "layout", str(run.root / "logs" / "worker-0.log")):
        assert part in error, (part, error)
    assert error in run.stderr or error.split("] ", 1)[-1] in run.stderr


def test_a_dead_local_node_fails_the_run(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="deadnode", overrides=FOREVER,
                    env={"COLOSSEUM_TEST_KILL_NODE_AFTER": "20"})
    assert run.returncode == 1, run.stderr[-3000:]
    assert f"node died (exit 9), see {run.root / 'logs' / 'node.log'}" in run.stderr
    assert json.loads((run.root / "run_state.json").read_text())["status"] == "failed"


def test_hub_without_a_config_is_a_usage_error(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "colosseum", "hub"], cwd=REPO_ROOT, env=child_env(),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "--config" in proc.stderr
