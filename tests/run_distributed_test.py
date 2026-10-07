"""End-to-end distributed (gRPC) smoke test (C2).

Spins up, on localhost:
  - a WeightStore gRPC server (in-process),
  - a learner process (run_distributed_learner: TrajectoryService + training +
    weight push to the store),
  - rollout workers (run_distributed_workers) that send chunks to the learner
    and pull weights from the store.

Verifies the learner actually trained and published weights (policy_version > 0)
to the store — i.e. the gRPC data + weight planes are really wired.
"""
import multiprocessing as mp
import os
import socket
import sys
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    from colosseum.distributed import run_distributed_learner, run_distributed_workers
    from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    ws_port = _free_port()
    traj_port = _free_port()
    ws_addr = f"localhost:{ws_port}"
    learner_addr = f"localhost:{traj_port}"
    agent = "agent_0"

    overrides = {
        "training.total_timesteps": 1200,
        "rollout.num_workers": 2,
        "rollout.envs_per_worker": 4,
        "rollout.chunk_length": 8,
        "rollout.weight_sync_interval_sec": 0.5,
        "learner.batch_chunks": 2,
        "learner.queue_size": 32,
        "self_play.checkpoint_interval": 1,  # exercise checkpoint drain thread
        "metrics.use_wandb": False,
    }
    cfg_path = "configs/examples/tic_tac_toe.yaml"

    # 1) Weight store server (background threads in this process).
    ws_server = serve_weight_store(port=ws_port)
    time.sleep(0.3)

    # 2) Learner process.
    learner = mp.Process(
        target=run_distributed_learner,
        args=(cfg_path, agent, traj_port, ws_addr, overrides),
        daemon=False,
    )
    learner.start()
    time.sleep(2.0)  # let the TrajectoryService bind

    # 3) Workers (blocks until total_timesteps reached).
    print("Starting distributed workers...", flush=True)
    run_distributed_workers(cfg_path, ws_addr, {agent: learner_addr}, overrides)
    print("Workers done.", flush=True)

    # Give the learner a moment to drain remaining chunks, then stop it.
    time.sleep(2.0)
    learner.terminate()
    learner.join(timeout=10)

    # 4) Verify weights were published.
    client = GRPCWeightStore(ws_addr)
    version = client.get_version(agent)
    payload = client.get(agent)
    client.close()
    ws_server.stop(0)

    print(f"Final published policy_version for {agent}: {version}", flush=True)
    assert payload is not None, "no weights published to the store"
    assert version > 0, f"learner did not train/publish (version={version})"
    print("Distributed gRPC pipeline OK!", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
