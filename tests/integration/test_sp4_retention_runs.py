"""Retention in a real run (SP4 T4.1): top-k with live ratings and the cap; every snapshot names its run."""
from __future__ import annotations

import json

from cli_runner import TTT_CONFIG, run_train
from colosseum.core.run_dir import RunDir


def test_a_top_k_run_stays_under_max_snapshots_and_writes_run_id(tmp_path):
    """``top_k_min_games: 10`` protects new snapshots until they have rated games, so the pool holds more than
    ``keep_last`` (the snapshot that just left the keep_last window has had no time to play 10 rated games) and
    reaches the cap; the cap still bounds it."""
    run = run_train(TTT_CONFIG, tmp_path, name="topk", overrides={
        "checkpoint.interval": "5", "checkpoint.keep_last": "2", "checkpoint.keep_every": "0",
        "checkpoint.keep_top_k": "2", "checkpoint.top_k_min_games": "10", "checkpoint.max_snapshots": "4",
        "ratings.refit_interval_sec": "0.5"})
    assert run.returncode == 0, run.stderr[-3000:]
    agent_dir = run.root / "checkpoints" / "agent_0"
    snapshots = sorted(agent_dir.glob("ckpt_v*"))
    assert 3 <= len(snapshots) <= 4, [p.name for p in snapshots]
    run_id = RunDir.open(run.root).run_id
    metas = [json.loads((p / "meta.json").read_text()) for p in snapshots]
    assert all(m["run_id"] == run_id and m["pinned"] is False for m in metas)
    assert sum(1 for m in metas if m.get("final") is True) == 1
    log = run.log("main")
    assert "Traceback" not in log and "ERROR" not in log and "exceed checkpoint.max_snapshots" not in log
