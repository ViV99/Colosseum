"""A hub killed with SIGKILL and resumed in place while its node lives (SP5 T4.3, spec 5.4 / 7.1, criterion 11's hub row
on localhost): the node reconnects to the resumed hub (same node id and number) through a TCP proxy whose target
moves to the new port, its learner is adopted (same generation), versions continue above every version accepted
before the kill, the results the node buffered arrive once (journal uids are unique; resent duplicates are counted),
cells granted after the resume start at the next epoch block, and the resumed hub exits 0 at its budget."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import colosseum.node.session as node_session
from cli_runner import REPO_ROOT, TTT_CONFIG, child_env, hub_cmd, wait_for, wait_for_hub_json
from cluster_kit import CLUSTER_TINY, journal_rows, read_hub_state, start_node
from colosseum.hub.registry import read_hub_json
from colosseum.net.security import HubIdentity
from fault_proxy import TcpFaultProxy

pytestmark = pytest.mark.slow

_EPOCH_RE = re.compile(r"_g(\d+)_")
AGENT = "agent_0"


def _start(cmd: list[str], out_name: str, tmp_path) -> subprocess.Popen:
    with open(tmp_path / f"{out_name}.stdout", "w") as out, open(tmp_path / f"{out_name}.stderr", "w") as err:
        return subprocess.Popen(cmd, cwd=REPO_ROOT, env=child_env(), stdout=out, stderr=err, text=True,
                                start_new_session=True)


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _port(address: str) -> int:
    return int(address.rsplit(":", 1)[1])


def _latest_versions(rows) -> list[int]:
    return [v for row in rows for s in row.seats if s.agent_id == AGENT and s.network_id == "latest"
            for v in (s.version_start, s.version_end) if v is not None]


def _epochs(rows) -> list[int]:
    return [int(m.group(1)) for row in rows if (m := _EPOCH_RE.search(row.uid))]


def test_kill_the_hub_and_resume_in_place_with_a_live_node(tmp_path, monkeypatch):
    monkeypatch.setattr(node_session, "RECONNECT_MIN_SEC", 0.05)
    monkeypatch.setattr(node_session, "RECONNECT_MAX_SEC", 0.2)
    overrides = {**CLUSTER_TINY, "training.total_timesteps": "40000", "distributed.hub_lost_collect_sec": "120",
                 "distributed.hub_lost_timeout_sec": "240", "distributed.node_lost_after_sec": "20",
                 "distributed.learner_relocate_after_sec": "60"}
    runs = tmp_path / "runs"
    first = _start(hub_cmd(TTT_CONFIG, runs, "r", overrides), "hub1", tmp_path)
    resumed = None
    node = None
    proxy = None
    try:
        root = runs / "r"
        info = wait_for_hub_json(root, 120)
        token = HubIdentity.load_or_create(Path(info.identity_dir)).token
        proxy = TcpFaultProxy(("127.0.0.1", _port(info.addresses[0])))
        hub = SimpleNamespace(address=proxy.address, token=token, insecure=True)
        node = start_node(tmp_path, hub, "n1", workers=1, cpu_learners=1, once=True)
        assert wait_for(lambda: (root / "hub_state.json").exists()
                        and (read_hub_state(root)["versions"].get(AGENT) or {}).get("max_version", -1) >= 2
                        and len(journal_rows(root)) >= 200, 180), (tmp_path / "hub1.stderr").read_text()[-3000:]
        before_state = read_hub_state(root)
        generation = before_state["learners"]["agents"][AGENT]["generation"]
        _kill(first)                                          # the node lives on
        rows_before = journal_rows(root)
        accepted_before = max(_latest_versions(rows_before) + [before_state["versions"][AGENT]["max_version"]])
        epochs_before = _epochs(rows_before)
        time.sleep(3.0)
        assert node.thread.is_alive()

        cmd = [sys.executable, "-m", "colosseum", "hub", "--resume", str(root), "--port", "0", "--insecure",
               "--listen", "127.0.0.1"]
        old_pid = info.pid
        resumed = _start(cmd, "hub2", tmp_path)
        assert wait_for(lambda: (i := read_hub_json(root)) is not None and i.pid != old_pid, 120)
        proxy.set_target(("127.0.0.1", _port(read_hub_json(root).addresses[0])))
        assert wait_for(lambda: read_hub_state(root)["learners"]["agents"][AGENT]["status"] == "running"
                        and read_hub_state(root)["env_steps"] > before_state["env_steps"], 120)
        assert read_hub_state(root)["learners"]["agents"][AGENT]["generation"] == generation   # adopted
        assert resumed.wait(300) == 0, (tmp_path / "hub2.stderr").read_text()[-3000:]
        assert node.stop() == 0
        rows = journal_rows(root)
        after = [r for r in rows if r.id > max(r.id for r in rows_before)]
        # R7: cells the resumed hub granted start at the next epoch block; the rest are the first hub's cells whose
        # results the node buffered over the gap
        reserved = before_state["epoch_reserved_until"]
        fresh = [e for e in _epochs(after) if e > reserved]
        assert fresh and min(fresh) == reserved + 1 and max(epochs_before) <= reserved
        new_versions = [v for row in after for s in row.seats if s.agent_id == AGENT and s.network_id == "latest"
                        and s.version_start is not None and s.version_start > accepted_before
                        for v in (s.version_start,)]
        assert new_versions, "no version above the ones accepted before the kill"
        assert len({r.uid for r in rows}) == len(rows)
        state = json.loads((root / "hub_state.json").read_text())
        totals = state["counters"]["book"].get("totals", {})
        assert totals.get("dropped_match_results_by_reason.node_buffer", 0) == 0
        log = (root / "logs" / "main.log").read_text()
        assert "Resume: hub state restored" in log and "Resume: replayed" in log
    finally:
        for proc in (first, resumed):
            if proc is not None:
                _kill(proc)
        if node is not None and node.thread.is_alive():
            node.agent.request_stop()
            node.thread.join(60)
        if proxy is not None:
            proxy.close()
