"""SP2 compatibility baseline (SP3 T0.1): `colosseum train` resumes from the SP2 fixture run dir."""
from __future__ import annotations

from cli_runner import run_train
from game_helpers import SP2_CHECKPOINT_ENV_STEPS, SP2_CHECKPOINT_VERSION, SP2_TTT_TINY, copy_sp2_run


def _versions(agent_dir) -> list[int]:
    return sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.glob("ckpt_v*"))


def test_training_resumes_from_the_sp2_run_dir(tmp_path):
    source = copy_sp2_run(tmp_path)
    # tiny=False: the fixture config is already tiny, and cli_runner.TINY's new-style keys (checkpoint.keep_last
    # from T2.1 on) next to the config's SP2 knobs (pool_size) would be "old + new together", a ConfigError.
    run = run_train(SP2_TTT_TINY, tmp_path, name="resumed", overrides={"training.resume_from": str(source)},
                    tiny=False)
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.log("main")
    assert "Resume [agent_0]:" in log and f"(policy_version {SP2_CHECKPOINT_VERSION})" in log
    assert f"env-step counter continues from {SP2_CHECKPOINT_ENV_STEPS}" in log
    agent_dir = run.root / "checkpoints" / "agent_0"
    versions = _versions(agent_dir)
    assert versions[0] == SP2_CHECKPOINT_VERSION and len(versions) >= 2   # the SP2 snapshot joined the new pool
    imported = agent_dir / f"ckpt_v{SP2_CHECKPOINT_VERSION}"
    assert sorted(p.name for p in imported.iterdir()) == ["meta.json", "model.pt"]   # final: kept; no trainer state
    assert "snapshot pool carried over" in log
    assert _versions(source / "checkpoints" / "agent_0") == [SP2_CHECKPOINT_VERSION]  # the source is untouched
