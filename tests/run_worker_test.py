"""Quick test: spawn a single worker and check if it produces chunks."""
import multiprocessing as mp
import sys
import os
import time
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import _worker_target

    config = load_config("configs/examples/tic_tac_toe.yaml")
    cd = config.model_dump()
    cd["training"]["total_timesteps"] = 2000
    cd["rollout"]["num_workers"] = 1
    cd["rollout"]["envs_per_worker"] = 2
    cd["rollout"]["chunk_length"] = 8
    cd["learner"]["batch_chunks"] = 2
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    agent_id = "agent_0"
    agent_configs = {agent_id: config}
    trajectory_queues = {agent_id: mp.Queue(maxsize=16)}
    weight_queues = {agent_id: mp.Queue(maxsize=2)}
    stop_event = mp.Event()

    print("Spawning worker...", flush=True)
    p = mp.Process(
        target=_worker_target,
        args=(0, config, [agent_id], agent_configs,
              trajectory_queues, weight_queues, stop_event, 200),
        daemon=True,
    )
    p.start()
    print(f"Worker pid={p.pid}", flush=True)

    chunks = 0
    for i in range(3):
        try:
            chunk = trajectory_queues[agent_id].get(timeout=10)
            chunks += 1
            print(f"Got chunk {i}: obs={chunk.observations.shape}, ver={chunk.behavior_policy_version}", flush=True)
        except Exception as e:
            print(f"Timeout on chunk {i}: {e}", flush=True)
            break

    stop_event.set()
    p.join(timeout=5)
    if p.is_alive():
        p.terminate()
    print(f"Worker done. Chunks received: {chunks}", flush=True)
    assert chunks > 0, "No chunks produced!"


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
