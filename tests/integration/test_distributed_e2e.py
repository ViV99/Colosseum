"""End-to-end distributed (gRPC) run on localhost: weight store + learner + workers."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time

import pytest

from helpers import example_config

pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_distributed_grpc_pipeline(tmp_path):
    """The learner trains on chunks from gRPC workers and publishes weights (version > 0)."""
    from colosseum.distributed import run_distributed_learner, run_distributed_workers
    from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    ws_port, traj_port = _free_port(), _free_port()
    ws_addr, learner_addr = f"localhost:{ws_port}", f"localhost:{traj_port}"
    agent = "agent_0"
    cfg_path = str(example_config("tic_tac_toe.yaml"))
    overrides = {
        "training.total_timesteps": 1200,
        "rollout.num_workers": 2,
        "rollout.envs_per_worker": 4,
        "rollout.chunk_length": 8,
        "rollout.weight_sync_interval_sec": 0.5,
        "learner.batch_chunks": 2,
        "learner.queue_size": 32,
        "self_play.checkpoint_interval": 1,
        "metrics.use_wandb": False,
        "checkpoint.dir": str(tmp_path / "checkpoints"),
    }

    ws_server = serve_weight_store(port=ws_port)
    learner = mp.Process(
        target=run_distributed_learner,
        args=(cfg_path, agent, traj_port, ws_addr, overrides),
        daemon=False,
    )
    learner.start()
    try:
        time.sleep(2.0)  # let the TrajectoryService bind
        run_distributed_workers(cfg_path, ws_addr, {agent: learner_addr}, overrides)
        time.sleep(2.0)  # let the learner drain the last chunks
    finally:
        learner.terminate()
        learner.join(timeout=10)

    client = GRPCWeightStore(ws_addr)
    try:
        version = client.get_version(agent)
        payload = client.get(agent)
    finally:
        client.close()
        ws_server.stop(0)
    assert payload is not None, "no weights were published to the store"
    assert version > 0, f"learner did not train/publish (version={version})"
