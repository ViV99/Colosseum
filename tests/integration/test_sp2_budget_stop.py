"""The hub (SP2's launcher, SP5 T2.5 b) stops every process once the global env-step budget is reached (T7.1)."""
import json
import threading
import time
from pathlib import Path

from cluster_kit import read_hub_state, wait_until
from colosseum.core.config import ColosseumConfig, load_config
from game_helpers import make_test_run_dir
from hub_kit import make_hub

REPO = Path(__file__).resolve().parents[2]


def test_launch_stops_at_the_env_step_budget(tmp_path):
    data = load_config(REPO / "configs/examples/tic_tac_toe.yaml").model_dump()
    data["training"]["total_timesteps"] = 400
    data["rollout"]["num_workers"] = 1
    data["rollout"]["envs_per_worker"] = 2
    data["rollout"]["chunk_length"] = 8
    data["learner"]["batch_chunks"] = 2
    data["learner"]["queue_size"] = 16
    data["learner"]["device"] = "cpu"
    data["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**data)
    hub = make_hub(config, make_test_run_dir(config, tmp_path))
    start = time.monotonic()
    assert hub.launch() == 0
    assert hub.env_steps_done >= 400
    assert time.monotonic() - start < 120


def test_a_learner_that_keeps_dying_fails_its_agent_and_the_other_trains_on(tmp_path, caplog):
    """Learners run until the run stops. SP5 T4.2 (spec 7.1, W04-25; it was "one exiting early stops the run" until
    T4.2, A12): agent_beta's learner dies at every start, so the hub restarts it (the node's ``LEARNER_RESTARTS``: 3)
    and then marks the agent ``failed``; agent_alpha keeps training (the workers never block on the dead learner); the
    run then ends with exit code 1 (a failed agent, spec 7.5). Controller ruling R41: with the example's rivals-only mix
    the failed agent (no weights) is no candidate, so alpha plays itself (the ``fallback`` source), never stalls."""
    data = load_config(REPO / "configs/examples/tic_tac_toe_multi.yaml").model_dump()
    data["training"]["total_timesteps"] = 10**9
    data["rollout"]["num_workers"] = 1
    data["rollout"]["envs_per_worker"] = 2
    data["learner"]["device"] = "cpu"
    data["metrics"]["use_wandb"] = False
    data["metrics"]["console_interval_sec"] = 0.5
    broken = dict(data["algorithm"], algorithm_class="colosseum.algorithms.appo.NoSuchAlgorithm")
    data["agents"]["agent_beta"]["algorithm"] = broken  # agent_beta's learner dies at startup
    # the example's rivals-only mix stays: beta never published weights, so once it failed the matchmaker leaves it
    # out (R41) and alpha's rivals share falls back to its own latest weights
    config = ColosseumConfig(**data)
    run = make_test_run_dir(config, tmp_path)
    hub = make_hub(config, run)
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(hub.launch()), name="hub", daemon=True)

    def learner(agent: str) -> dict:
        return (read_hub_state(run.root).get("learners") or {}).get("agents", {}).get(agent) or {}

    def alpha_steps() -> int:
        if not run.metrics_path.exists():
            return 0
        records = [json.loads(line) for line in run.metrics_path.read_text().splitlines() if line.strip()]
        return max((r["train_step"] for r in records if r["kind"] == "train" and r["agent"] == "agent_alpha"),
                   default=0)

    with caplog.at_level("ERROR", logger="colosseum.hub.hub"):
        thread.start()
        try:
            assert wait_until(lambda: learner("agent_beta").get("status") == "failed", 120)
            steps = alpha_steps()
            assert wait_until(lambda: alpha_steps() > steps, 60)          # alpha trains on
            assert learner("agent_alpha")["status"] == "running"
        finally:
            hub.request_stop()
            thread.join(60)
    assert not thread.is_alive() and codes == [1]
    assert hub.env_steps_done < 10**9
    # The failure names the dead process's reason and points to its log (T6.5); the exit names the failed agent.
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    (failed,) = [e for e in errors if e.startswith("Learner of agent_beta failed")]
    assert "NoSuchAlgorithm" in failed and f"see {run.logs / 'learner-agent_beta.log'}" in failed
    assert any("Training of agent_beta failed" in e for e in errors)
    # The crash and its traceback are in the dead learner's own log file (T6.2), once per incarnation.
    crash_log = (run.logs / "learner-agent_beta.log").read_text()
    assert "learner-agent_beta crashed" in crash_log and crash_log.count("NoSuchAlgorithm") >= 4
