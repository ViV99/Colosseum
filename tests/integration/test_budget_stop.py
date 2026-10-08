"""The launcher stops every process once the global env-step budget is reached (T2.5)."""
import time
from pathlib import Path

from colosseum.core.config import ColosseumConfig, load_config
from colosseum.launcher import Launcher
from helpers import make_test_run_dir

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
    launcher = Launcher(config, make_test_run_dir(config, tmp_path))
    start = time.monotonic()
    assert launcher.launch() == 0
    assert launcher.env_steps_done >= 400
    assert time.monotonic() - start < 120


def test_a_dead_learner_stops_the_run(tmp_path, caplog):
    """Local learners run until stop_event; one exiting early must stop the run.

    Otherwise the workers block on the dead learner's full chunk queue and the
    other agent starves while the budget is never reached.
    """
    data = load_config(REPO / "configs/examples/tic_tac_toe_multi.yaml").model_dump()
    data["training"]["total_timesteps"] = 10**9
    data["rollout"]["num_workers"] = 1
    data["rollout"]["envs_per_worker"] = 2
    data["learner"]["device"] = "cpu"
    data["metrics"]["use_wandb"] = False
    broken = dict(data["algorithm"], algorithm_class="colosseum.algorithms.appo.NoSuchAlgorithm")
    data["agents"]["agent_beta"]["algorithm"] = broken  # agent_beta's learner dies at startup
    config = ColosseumConfig(**data)
    run = make_test_run_dir(config, tmp_path)
    launcher = Launcher(config, run)
    start = time.monotonic()
    with caplog.at_level("ERROR", logger="colosseum.launcher"):
        assert launcher.launch() == 1
    assert time.monotonic() - start < 120
    assert launcher.env_steps_done < 10**9
    # The failure names the dead process and points to its log (T6.5).
    expected = f"learner-agent_beta died (exit 1), see {run.logs / 'learner-agent_beta.log'}"
    assert any(r.levelname == "ERROR" and r.getMessage() == expected for r in caplog.records)
    # The crash and its traceback are in the dead learner's own log file (T6.2).
    crash_log = (run.logs / "learner-agent_beta.log").read_text()
    assert "learner-agent_beta crashed" in crash_log and "NoSuchAlgorithm" in crash_log
