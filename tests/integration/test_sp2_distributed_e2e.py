"""End-to-end distributed (gRPC) self-play on localhost: weight store + learner + workers (T6.4)."""
from __future__ import annotations

import json
import multiprocessing as mp
import socket
import time

import pytest
import torch

from game_helpers import write_test_config

pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for(condition, timeout: float, what: str) -> None:
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
def test_distributed_grpc_pipeline(tmp_path, restore_root_logging):
    """The learner trains on chunk v2 from gRPC workers, publishes weights and saves signed checkpoints."""
    from colosseum.sp2.distributed import run_distributed_learner, run_distributed_workers, workers_role
    from colosseum.sp2.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    ws_port, traj_port = _free_port(), _free_port()
    ws_addr, learner_addr = f"localhost:{ws_port}", f"localhost:{traj_port}"
    agent = "agent_0"
    cfg_path = str(write_test_config(tmp_path / "turns.yaml", "turns"))
    overrides = {
        "training.total_timesteps": 1200, "rollout.num_workers": 2, "rollout.envs_per_worker": 4,
        "rollout.chunk_length": 8, "rollout.weight_sync_interval_sec": 0.5, "learner.batch_chunks": 2,
        "learner.queue_size": 32, "checkpoint.interval": 1, "run.dir": str(tmp_path / "runs"), "run.name": "e2e",
    }
    learner_run = tmp_path / "runs" / f"e2e-learner-{agent}"
    workers_run = tmp_path / "runs" / f"e2e-{workers_role()}"
    ckpt_root = learner_run / "checkpoints" / agent
    ws_server = serve_weight_store(port=ws_port)
    learner = mp.Process(target=run_distributed_learner, args=(cfg_path, agent, traj_port, ws_addr, overrides),
                         daemon=False)
    learner.start()
    client = GRPCWeightStore(ws_addr)
    try:
        _wait_for(lambda: _port_open(traj_port), 60, "the learner's TrajectoryService to bind")
        assert run_distributed_workers(cfg_path, ws_addr, {agent: learner_addr}, overrides) == 0
        _wait_for(lambda: client.get_version(agent) > 0, 60, "a trained weight version in the store")
        _wait_for(lambda: any(ckpt_root.glob("ckpt_v*/meta.json")), 60, "a checkpoint saved by the learner")
        version = client.get_version(agent)
    finally:
        learner.terminate()
        learner.join(timeout=10)
        if learner.is_alive():
            learner.kill()
            learner.join()
        client.close()
        ws_server.stop(0)
    assert version > 0
    newest = max((m.parent for m in ckpt_root.glob("ckpt_v*/meta.json")), key=lambda d: int(d.name[len("ckpt_v"):]))
    assert all(isinstance(v, torch.Tensor) for v in torch.load(newest / "model.pt", weights_only=True).values())
    meta = json.loads((newest / "meta.json").read_text())
    assert meta["roles"] and isinstance(meta["role_signature"], str) and meta["env_steps"] is None
    for run in (learner_run, workers_run):
        assert (run / "config.resolved.yaml").is_file()
    for worker_id in range(2):
        log = (workers_run / "logs" / f"worker-{worker_id}.log").read_text()
        assert f"worker-{worker_id} started (pid" in log and f"worker-{worker_id} finished" in log
