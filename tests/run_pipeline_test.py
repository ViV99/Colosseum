"""Full pipeline integration test."""
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
    cd["training"]["total_timesteps"] = 3000
    cd["rollout"]["num_workers"] = 1
    cd["rollout"]["envs_per_worker"] = 2
    cd["rollout"]["chunk_length"] = 8
    cd["learner"]["batch_chunks"] = 2
    cd["learner"]["queue_size"] = 16
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    print("Starting full pipeline...", flush=True)
    launcher = Launcher(config)
    launcher.launch()
    print("Pipeline completed!", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
