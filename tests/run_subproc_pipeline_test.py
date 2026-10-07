"""End-to-end pipeline smoke test with the subprocess vec_env backend (C12).

Runs the full Launcher with rollout.vec_env="subprocess" to verify nested
spawn (worker process -> env subprocesses) works and training completes.
"""
import multiprocessing as mp
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import Launcher

    config = load_config("configs/examples/tic_tac_toe.yaml")
    cd = config.model_dump()
    cd["training"]["total_timesteps"] = 2000
    cd["rollout"]["num_workers"] = 1
    cd["rollout"]["envs_per_worker"] = 4
    cd["rollout"]["chunk_length"] = 8
    cd["rollout"]["vec_env"] = "subprocess"
    cd["rollout"]["subproc_workers"] = 2
    cd["rollout"]["match_refresh_interval_sec"] = 1.0
    cd["learner"]["batch_chunks"] = 2
    cd["learner"]["queue_size"] = 16
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    print("Starting subprocess-vec_env pipeline...", flush=True)
    launcher = Launcher(config)
    launcher.launch()
    print("Subprocess pipeline completed!", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
