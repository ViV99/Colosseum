"""SP3 compatibility baseline (SP4 T0.1): `colosseum train` resumes from the SP3 run fixture (a copy) and carries
its snapshot pool over; the source is only read. T1.3: the new run starts an empty journal (INFO); T3.2: an empty
league state (INFO)."""
from __future__ import annotations

from cli_runner import run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from game_helpers import SP3_TTT_TINY, copy_sp3_run, sp3_manifest, tree_digest


def _versions(agent_dir) -> list[int]:
    return sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))


def test_training_resumes_from_the_sp3_run_dir(tmp_path):
    manifest = sp3_manifest()
    source = copy_sp3_run(tmp_path)
    before = tree_digest(source)
    budget = manifest["env_steps"] + 2000          # a resume needs a budget above the checkpoint's env_steps
    run = run_train(SP3_TTT_TINY, tmp_path, name="resumed",
                    overrides={"training.resume_from": str(source), "training.total_timesteps": str(budget)})
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.log("main")
    assert f"Resume [{manifest['agent']}]:" in log and f"(policy_version {manifest['final_version']})" in log
    assert f"env-step counter continues from {manifest['env_steps']}" in log
    assert "snapshot pool carried over" in log
    versions = _versions(run.root / "checkpoints" / manifest["agent"])
    assert min(versions) <= manifest["final_version"] < max(versions)   # imported snapshots + new ones
    assert tree_digest(source) == before        # the source is untouched: names and contents (hard links)
    assert f"has no {MATCH_LOG_FILE} (an SP3 run dir); the journal starts empty" in log        # T1.3
    reader = MatchLogReader(run.root / MATCH_LOG_FILE)
    (only,) = reader.runs()
    assert only.parent_run_id is None                                     # an SP3 run dir has no run id
    assert reader.matches(limit=1) and all(r.run_id == only.run_id for r in reader.matches(limit=1000))
    reader.close()
    assert "has no league_state.json (an SP3 run dir); ratings and PFSP statistics start empty" in log   # T3.2
    assert (run.root / "league_state.json").is_file()
