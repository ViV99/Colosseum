"""Smoke: `colosseum eval` loads checkpoints written by a real `colosseum train` run (T7.2)."""
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
    assert "networks" in json.loads((ckpts[-1] / "meta.json").read_text())

    out = tmp_path / "out.json"
    proc = run_in_session([
        sys.executable, "-m", "colosseum", "eval", "-c", str(TTT_CONFIG),
        "-a", f"x={ckpts[0]}", "-a", f"y={ckpts[-1]}", "-n", "4", "--num-envs", "2", "--seed", "0",
        "-o", str(out),
    ], timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["mode"] == "pairwise" and data["agents"] == ["x", "y"]
    assert [(r["agent_a"], r["agent_b"]) for r in data["pairs"]] == [("x", "y"), ("y", "x")]
    assert all(r["n"] == 4 and r["wins"] + r["draws"] + r["losses"] == 4 for r in data["pairs"])
