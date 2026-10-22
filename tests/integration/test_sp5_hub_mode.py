"""``colosseum hub``: a hub process that nodes join (SP5 T2.5; spec 1, 4.1-4.4; amendment A11; pre-flight ruling
P-PF2-16): a node in the test process registers, gets the learner (the first node with learner capacity) and its
worker's lineups, and the hub stops it at the budget."""
from __future__ import annotations

import os
import re
import signal
from pathlib import Path

from cli_runner import TINY, TTT_CONFIG, start_hub, wait_for_hub_json
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.net.security import HubIdentity
from colosseum.node.agent import NodeOptions
from node_kit import NodeThread


def test_hub_trains_with_a_node_that_connects_in_process(tmp_path):
    proc, root = start_hub(TTT_CONFIG, tmp_path / "runs", "hubrun", dict(TINY), port=0, insecure=True)
    node = None
    try:
        info = wait_for_hub_json(root, 120)
        assert info.mode == "hub" and info.tls is False and re.fullmatch(r"127\.0\.0\.1:[1-9]\d*", info.loopback)
        identity = HubIdentity.load_or_create(Path(info.identity_dir))
        node = NodeThread(NodeOptions(hub_address=info.loopback, token=identity.token, insecure=True, workers=1,
                                      cpu_learners=1, identity_root=tmp_path / "node",
                                      log_dir=tmp_path / "node-logs"))
        assert proc.wait(240) == 0, (root.parent / "hubrun.stderr").read_text()[-3000:]
        node.thread.join(60)
        assert not node.thread.is_alive() and node.exit_code == 0
    finally:
        if node is not None and node.thread.is_alive():
            node.stop()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    main = (root / "logs" / "main.log").read_text()
    assert "Learners of ['agent_0'] placed on node 1" in main and "Training budget reached" in main
    assert (tmp_path / "node-logs" / "learner-agent_0.log").exists()
    with MatchLogReader(root / MATCH_LOG_FILE) as reader:
        uids = [r.uid for r in reader.matches(source="train", limit=None)]
    assert uids and all("/train/n1/" in uid for uid in uids)
    assert any((root / "checkpoints" / "agent_0").glob("ckpt_v*"))


def test_a_hub_port_that_is_taken_is_a_config_error(tmp_path):
    import socket

    from cli_runner import hub_cmd, run_in_session

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        proc = run_in_session(hub_cmd(TTT_CONFIG, tmp_path / "runs", "taken", dict(TINY), port=port), timeout=120)
    assert proc.returncode == 1 and "Config error: the hub cannot listen" in proc.stderr, proc.stderr[-2000:]
    assert "Traceback" not in proc.stderr
