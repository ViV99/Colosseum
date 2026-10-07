"""End-to-end runs through real worker / learner processes (spawn start method)."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import pytest

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import ColosseumConfig, load_config
from helpers import example_config


def _config(name: str, tmp_path, **sections: dict) -> ColosseumConfig:
    """Example config with checkpoints under tmp_path, WandB off and section overrides."""
    data = load_config(example_config(name)).model_dump()
    data["metrics"]["use_wandb"] = False
    data["checkpoint"]["dir"] = str(tmp_path / "checkpoints")
    for section, values in sections.items():
        data[section].update(values)
    return ColosseumConfig(**data)


def _stop(proc: mp.Process, stop_event) -> None:
    stop_event.set()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)


def test_worker_produces_chunks(tmp_path):
    """A spawned worker process sends full-length chunks for its agent."""
    from colosseum.launcher import _worker_target

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
    )
    agent_id = "agent_0"
    trajectory_queues = {agent_id: mp.Queue(maxsize=16)}
    weight_queues = {agent_id: mp.Queue(maxsize=2)}
    stop_event = mp.Event()
    proc = mp.Process(
        target=_worker_target,
        args=(0, config, [agent_id], {agent_id: config}, trajectory_queues, weight_queues, stop_event, 500),
        daemon=True,
    )
    proc.start()
    try:
        chunks = [trajectory_queues[agent_id].get(timeout=60) for _ in range(5)]
    finally:
        _stop(proc, stop_event)
    for chunk in chunks:
        assert chunk.agent_id == agent_id
        assert chunk.observations.shape[0] == 8


def test_worker_multi_agent_routing(tmp_path):
    """Chunks are routed to the queue of the agent that occupies the slot."""
    from colosseum.launcher import _worker_target

    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
    )
    agent_ids = config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}
    trajectory_queues = {aid: mp.Queue(maxsize=16) for aid in agent_ids}
    weight_queues = {aid: mp.Queue(maxsize=2) for aid in agent_ids}
    stop_event = mp.Event()
    slot_agent_map = [[agent_ids[0], agent_ids[1]], [agent_ids[0], agent_ids[1]]]
    slot_network_map = [["latest", "latest"], ["latest", "latest"]]
    collect_mask = [[True, True], [True, True]]
    proc = mp.Process(
        target=_worker_target,
        args=(
            0, config, agent_ids, agent_configs, trajectory_queues, weight_queues,
            stop_event, 500, None, slot_network_map, collect_mask, slot_agent_map, None,
        ),
        daemon=True,
    )
    proc.start()
    chunks_by_agent: dict[str, list] = {aid: [] for aid in agent_ids}
    deadline = time.time() + 60
    try:
        while time.time() < deadline and min(len(v) for v in chunks_by_agent.values()) < 2:
            for aid in agent_ids:
                try:
                    chunks_by_agent[aid].append(trajectory_queues[aid].get(timeout=0.5))
                except queue.Empty:
                    pass
    finally:
        _stop(proc, stop_event)
    for aid in agent_ids:
        assert len(chunks_by_agent[aid]) >= 2, f"agent {aid} received too few chunks"
        assert all(c.agent_id == aid for c in chunks_by_agent[aid])


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_full_pipeline(tmp_path):
    """Single-agent self-play training runs to completion and saves checkpoints."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    Launcher(config).launch()
    assert CheckpointManager(str(tmp_path / "checkpoints")).list_checkpoints("agent_0")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_full_pipeline_with_checkpoint_pool(tmp_path):
    """Checkpoints are saved every N train steps and the FIFO pool is respected."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 5000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        # 312 train steps: checkpoints at 40, 80, ..., 280 -> FIFO keeps the last 5.
        self_play={"checkpoint_interval": 40, "pool_size": 5},
    )
    Launcher(config).launch()
    ckpts = CheckpointManager(str(tmp_path / "checkpoints"), pool_size=5).list_checkpoints("agent_0")
    assert [c.policy_version for c in ckpts] == [120, 160, 200, 240, 280]


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_multi_agent_pipeline(tmp_path):
    """Two-agent league training runs to completion."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        self_play={"checkpoint_interval": 50},
    )
    Launcher(config).launch()
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    assert any(manager.list_checkpoints(aid) for aid in config.get_trainable_agent_ids())


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_subprocess_vec_env_pipeline(tmp_path):
    """Nested spawn (worker -> env subprocesses) trains to completion."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 2000},
        rollout={
            "num_workers": 1, "envs_per_worker": 4, "chunk_length": 8,
            "vec_env": "subprocess", "subproc_workers": 2, "match_refresh_interval_sec": 1.0,
        },
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    Launcher(config).launch()
    assert CheckpointManager(str(tmp_path / "checkpoints")).list_checkpoints("agent_0")
