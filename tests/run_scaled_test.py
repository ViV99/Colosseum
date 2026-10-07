"""Scaled test: 4 workers, more timesteps, verify weight sync works."""
import multiprocessing as mp
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import Launcher

    config = load_config("configs/examples/tic_tac_toe.yaml")
    cd = config.model_dump()
    cd["training"]["total_timesteps"] = 50000
    cd["rollout"]["num_workers"] = 4
    cd["rollout"]["envs_per_worker"] = 4
    cd["rollout"]["chunk_length"] = 16
    cd["rollout"]["weight_sync_interval_sec"] = 1.0
    cd["learner"]["batch_chunks"] = 8
    cd["learner"]["queue_size"] = 64
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    print("Starting scaled pipeline (4 workers, 50K steps)...", flush=True)
    launcher = Launcher(config)
    launcher.launch()
    print("Scaled test completed!", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
