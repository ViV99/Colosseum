"""SP4 compatibility baseline (SP5 T0.1, criterion 6): `colosseum train` with ``training.resume_from`` = a copy of the
SP4 run fixture starts a child run that continues the counters, carries the pool over (the pinned snapshot too),
restores the league state and imports the parent journal; the source is only read. T4.3 adds the in-place
``--resume`` of a copy (journal migration, same run id)."""
from __future__ import annotations

from cli_runner import run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.run_dir import read_run_id
from game_helpers import SP4_TTT_TINY, copy_sp4_run, sp4_manifest, tree_digest


def _digest(root) -> dict[str, str]:
    """``tree_digest`` without SQLite's -wal / -shm side files (opening a journal may create them)."""
    return {k: v for k, v in tree_digest(root).items() if not k.endswith(("-wal", "-shm"))}


def test_training_resumes_from_the_sp4_run_dir_as_a_child_run(tmp_path):
    manifest = sp4_manifest()
    source = copy_sp4_run(tmp_path)
    before = _digest(source)
    budget = manifest["env_steps"] + 2000
    run = run_train(SP4_TTT_TINY, tmp_path, name="child", tiny=False,
                    overrides={"training.resume_from": str(source), "training.total_timesteps": str(budget),
                               "evaluation.enabled": "false"})
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.log("main")
    assert f"Resume [{manifest['agent']}]:" in log and f"(policy_version {manifest['final_version']})" in log
    assert f"env-step counter continues from {manifest['env_steps']}" in log
    assert "league state (ratings, PFSP statistics) restored from" in log
    assert "snapshot pool carried over" in log
    copied = manifest["journal"]["by_source"]["pulse"] + manifest["journal"]["by_source"]["tournament"]
    assert f"{copied} tournament / pulse matches copied" in log
    assert f"{manifest['journal']['by_source']['train']} detailed training matches folded into counts" in log
    assert read_run_id(run.root) != manifest["run_id"]
    with MatchLogReader(run.root / MATCH_LOG_FILE) as reader:
        assert [r.run_id for r in reader.runs()][1:] == [manifest["run_id"]]
        assert len(reader.matches(source="tournament", limit=None)) == manifest["journal"]["by_source"]["tournament"]
    stored = sorted(int(p.name.removeprefix("ckpt_v")) for p in (run.root / "checkpoints" / "main").glob("ckpt_v*"))
    assert set(manifest["pinned"]) <= set(stored) and max(stored) > manifest["final_version"]
    assert _digest(source) == before
