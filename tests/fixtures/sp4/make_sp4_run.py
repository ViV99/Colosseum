"""Generator of the SP4 run-dir fixture (SP5 task T0.1). Run ONCE on unchanged SP4 code; the output is committed.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp4/make_sp4_run.py

Trains ``sp4_ttt_tiny.yaml`` with ``colosseum train`` in a temporary directory, pins the second-newest snapshot
(``colosseum pin``), plays a small gauntlet on the run (``colosseum tournament --run``) and copies what an SP4 run
dir holds into ``tests/fixtures/sp4/run/``: ``config.resolved.yaml`` (``run.dir`` rewritten to ``runs``), ``run.json``,
``run_state.json``, ``metrics.jsonl``, ``ratings.json``, ``matches.db`` (its WAL checkpointed into the file first),
``league_state.json``, ``pulse_state.json``, ``checkpoints/`` and ``tournaments/`` (every ``logs/`` is left out: they
hold machine paths). It then writes ``manifest.json`` with what tests need to know about the run: training is
asynchronous, so the snapshot versions and the match counts differ between generations. It is a record of how the
fixture was made; SP5 tasks never re-run it (the code after T1.1 makes SP5 run dirs).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG = HERE / "sp4_ttt_tiny.yaml"
AGENT = "main"
ANCHOR = "random"
TOURNAMENT = "t0"
FILES = ("config.resolved.yaml", "run.json", "run_state.json", "metrics.jsonl", "ratings.json", "matches.db",
         "league_state.json", "pulse_state.json")


def _colosseum(*args: str) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
    subprocess.run([sys.executable, "-m", "colosseum", *args], cwd=REPO_ROOT, env=env, check=True)


def _journal_counts(db: Path) -> dict:
    conn = sqlite3.connect(db)
    try:
        sources = dict(conn.execute("SELECT e.name, COUNT(*) FROM matches m JOIN entities e ON e.id = m.source "
                                    "GROUP BY e.name").fetchall())
        schema = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        counts = conn.execute("SELECT COUNT(*) FROM counts").fetchone()[0]
        runs = [r[0] for r in conn.execute("SELECT run_id FROM runs ORDER BY id")]
    finally:
        conn.close()
    return {"by_source": {k: int(v) for k, v in sorted(sources.items())}, "schema_version": int(schema),
            "counts_rows": int(counts), "run_ids": runs}


def main() -> None:
    dst = HERE / "run"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sp4_run"
        _colosseum("train", "-c", str(CONFIG), "--set", f"run.dir={tmp}", "--set", "run.name=sp4_run")
        versions = sorted(int(p.name.removeprefix("ckpt_v")) for p in (src / "checkpoints" / AGENT).glob("ckpt_v*"))
        assert len(versions) >= 3, versions
        pinned = versions[-2]
        _colosseum("pin", str(src), f"{AGENT}@v{pinned}")
        _colosseum("tournament", "--run", str(src), "--snapshots", "last:2", "--mode", "gauntlet", "--blocks", "2",
                   "--seed", "0", "--num-workers", "1", "--num-envs", "2", "--name", TOURNAMENT)
        conn = sqlite3.connect(src / "matches.db")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        assert not (src / "matches.db-wal").exists() or (src / "matches.db-wal").stat().st_size == 0
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        for name in FILES:
            shutil.copy2(src / name, dst / name)
        resolved = yaml.safe_load((src / "config.resolved.yaml").read_text())
        resolved["run"]["dir"] = "runs"
        (dst / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        shutil.copytree(src / "checkpoints", dst / "checkpoints")
        shutil.copytree(src / "tournaments", dst / "tournaments", ignore=shutil.ignore_patterns("logs"))
    agent_dir = dst / "checkpoints" / AGENT
    metas = {p.name: json.loads((p / "meta.json").read_text()) for p in sorted(agent_dir.iterdir())}
    finals = [m["policy_version"] for m in metas.values() if m["final"]]
    assert len(finals) == 1, finals
    run_id = json.loads((dst / "run.json").read_text())["run_id"]
    journal = _journal_counts(dst / "matches.db")
    assert journal["run_ids"] == [run_id] and set(journal["by_source"]) == {"train", "pulse", "tournament"}, journal
    manifest = {
        "agent": AGENT,
        "anchor": ANCHOR,
        "layout": "2p",
        "run_id": run_id,
        "versions": sorted(m["policy_version"] for m in metas.values()),
        "final_version": finals[0],
        "pinned": sorted(m["policy_version"] for m in metas.values() if m.get("pinned")),
        "with_trainer_state": sorted(m["policy_version"] for name, m in metas.items()
                                     if (agent_dir / name / "trainer_state.pt").exists()),
        "env_steps": max(int(m["env_steps"]) for m in metas.values()),
        "league_env_steps": json.loads((dst / "league_state.json").read_text())["env_steps"],
        "journal": {k: journal[k] for k in ("by_source", "schema_version", "counts_rows")},
        "tournament": TOURNAMENT,
        "files": sorted(p.name for p in dst.iterdir()),
    }
    assert manifest["pinned"] == [pinned] and len(manifest["with_trainer_state"]) < len(manifest["versions"])
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {dst}: {manifest}")


if __name__ == "__main__":
    main()
