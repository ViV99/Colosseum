"""A short SP2 `train` writes the run dir, the resolved config and per-process logs (T7.1)."""
from __future__ import annotations

from cli_runner import TTT_CONFIG, run_train
from colosseum.coordinator.checkpoint_manager import resolve_resume
from colosseum.core.config import load_config


def test_run_dir_has_resolved_config_and_process_logs(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="logs-run")
    assert run.returncode == 0, run.stderr[-3000:]
    assert f"Run directory: {run.root}" in run.stdout
    cfg = load_config(run.root / "config.resolved.yaml")
    assert cfg.run.name == "logs-run" and cfg.training.total_timesteps == 3000
    assert (run.root / "checkpoints").is_dir()
    assert "main started (pid" in run.log("main")
    learner = run.log("learner-agent_0")
    assert "learner-agent_0 started (pid" in learner and "learner-agent_0 finished" in learner
    worker = run.log("worker-0")
    assert "worker-0 started (pid" in worker and "worker-0 finished" in worker
    # The run dir is a valid resume source: the learner's checkpoints landed in it.
    assert resolve_resume(str(run.root), "agent_0")["policy_version"] > 0
