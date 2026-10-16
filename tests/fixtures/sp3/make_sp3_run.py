"""Generator of the SP3 run-dir fixture (SP4 task T0.1). Run ONCE on unchanged SP3 code; the output is committed.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp3/make_sp3_run.py

Trains ``sp3_ttt_tiny.yaml`` with ``colosseum train`` in a temporary directory and copies what an SP3 run dir
holds into ``tests/fixtures/sp3/run/``: ``config.resolved.yaml`` (``run.dir`` rewritten to ``runs``),
``metrics.jsonl``, ``ratings.json`` and ``checkpoints/`` (``logs/`` is left out: it holds machine paths). It then
writes ``manifest.json`` with what tests need to know about the run: training is asynchronous, so the snapshot
versions differ between generations. It is a record of how the fixture was made; SP4 tasks never re-run it (the
code after T1.1 makes SP4 run dirs).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG = HERE / "sp3_ttt_tiny.yaml"
AGENT = "main"
ANCHOR = "random"


def main() -> None:
    dst = HERE / "run"
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, "-m", "colosseum", "train", "-c", str(CONFIG),
               "--set", f"run.dir={tmp}", "--set", "run.name=sp3_run"]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
        src = Path(tmp) / "sp3_run"
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        for name in ("metrics.jsonl", "ratings.json"):
            shutil.copy2(src / name, dst / name)
        resolved = yaml.safe_load((src / "config.resolved.yaml").read_text())
        resolved["run"]["dir"] = "runs"
        (dst / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        shutil.copytree(src / "checkpoints", dst / "checkpoints")
    assert not (dst / "matches.db").exists() and not (dst / "league_state.json").exists()
    agent_dir = dst / "checkpoints" / AGENT
    metas = {p.name: json.loads((p / "meta.json").read_text()) for p in sorted(agent_dir.iterdir())}
    finals = [m["policy_version"] for m in metas.values() if m["final"]]
    assert len(finals) == 1, finals
    manifest = {
        "agent": AGENT,
        "anchor": ANCHOR,
        "layout": "2p",
        "versions": sorted(m["policy_version"] for m in metas.values()),
        "final_version": finals[0],
        "with_trainer_state": sorted(m["policy_version"] for name, m in metas.items()
                                     if (agent_dir / name / "trainer_state.pt").exists()),
        "env_steps": max(int(m["env_steps"]) for m in metas.values()),
    }
    assert len(manifest["versions"]) >= 3 and len(manifest["with_trainer_state"]) < len(manifest["versions"])
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {dst}: {manifest}")


if __name__ == "__main__":
    main()
