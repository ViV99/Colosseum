"""Smoke: `python -m colosseum eval` loads checkpoints written by a real SP2 `train` run (T7.1)."""
from __future__ import annotations

import json
import re
import sys

from cli_runner import TTT_CONFIG, run_in_session, run_train


def test_eval_loads_launcher_checkpoints(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="eval-smoke")
    assert run.returncode == 0, run.stderr[-2000:]
    agent_dir = run.root / "checkpoints" / "agent_0"
    ckpts = sorted((d for d in agent_dir.iterdir() if re.fullmatch(r"ckpt_v\d+", d.name)),
                   key=lambda d: int(d.name[len("ckpt_v"):]))
    assert ckpts, f"no checkpoints in {agent_dir}"
    meta = json.loads((ckpts[-1] / "meta.json").read_text())
    assert "networks" in meta and meta["roles"] == ["player"]

    out = tmp_path / "out.json"
    proc = run_in_session([
        sys.executable, "-m", "colosseum", "eval", "-c", str(TTT_CONFIG),
        "-a", f"x={ckpts[0]}", "-a", f"y={ckpts[-1]}", "-n", "4", "--num-envs", "2", "--seed", "0",
        "-o", str(out),
    ], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["agents"] == ["x", "y"] and list(data["layouts"]) == ["2p"]
    section = data["layouts"]["2p"]
    assert [(r["agent_a"], r["agent_b"]) for r in section["pairs"]] == [("x", "y"), ("y", "x")]
    assert all(r["n"] == 4 and r["wins"] + r["draws"] + r["losses"] == 4 for r in section["pairs"])
