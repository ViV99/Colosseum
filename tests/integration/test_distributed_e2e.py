"""End-to-end distributed (gRPC) run on localhost: weight store + learner + workers."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time

import pytest
import torch

from helpers import example_config

pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for(condition, timeout: float, what: str) -> None:
    """Poll ``condition()`` every 0.1 s until it is true; fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(0.1)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("localhost", port)) == 0


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

    ckpt_root = tmp_path / "checkpoints" / agent
    ws_server = serve_weight_store(port=ws_port)
    learner = mp.Process(
        target=run_distributed_learner,
        args=(cfg_path, agent, traj_port, ws_addr, overrides),
        daemon=False,
    )
    learner.start()
    client = GRPCWeightStore(ws_addr)
    try:
        _wait_for(lambda: _port_open(traj_port), 60, "the learner's TrajectoryService to bind")
        run_distributed_workers(cfg_path, ws_addr, {agent: learner_addr}, overrides)
        _wait_for(lambda: client.get_version(agent) > 0, 60, "a trained weight version in the store")
        # The learner's checkpoint drainer turns numpy snapshots back into torch files.
        _wait_for(lambda: any(ckpt_root.glob("*/meta.json")), 60, "a checkpoint saved by the learner")
        version = client.get_version(agent)
        payload = client.get(agent)
    finally:
        learner.terminate()
        learner.join(timeout=10)
        client.close()
        ws_server.stop(0)
    assert payload is not None, "no weights were published to the store"
    assert version > 0, f"learner did not train/publish (version={version})"
    # The newest checkpoint is complete (model.pt is written before meta.json) and
    # cannot have been evicted; the learner has exited, so nothing writes any more.
    newest = max((m.parent for m in ckpt_root.glob("*/meta.json")), key=lambda d: int(d.name[len("ckpt_v"):]))
    state_dict = torch.load(newest / "model.pt", weights_only=True)
    assert state_dict and all(isinstance(v, torch.Tensor) for v in state_dict.values())
