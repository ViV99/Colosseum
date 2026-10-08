"""End-to-end runs through real worker / learner processes (spawn start method)."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import pytest

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.types import TrajectoryChunk
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
        kwargs=dict(
            worker_id=0, config=config, agent_ids=[agent_id], agent_configs={agent_id: config},
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event,
        ),
        daemon=True,
    )
    proc.start()
    try:
        chunks = [TrajectoryChunk.from_payload(trajectory_queues[agent_id].get(timeout=60)) for _ in range(5)]
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
        kwargs=dict(
            worker_id=0, config=config, agent_ids=agent_ids, agent_configs=agent_configs,
            trajectory_queues=trajectory_queues, weight_queues=weight_queues,
            stop_event=stop_event,
            slot_network_map=slot_network_map, collect_mask=collect_mask, slot_agent_map=slot_agent_map,
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
                    chunks_by_agent[aid].append(
                        TrajectoryChunk.from_payload(trajectory_queues[aid].get(timeout=0.5)))
                except queue.Empty:
                    pass
    finally:
        _stop(proc, stop_event)
    for aid in agent_ids:
        assert len(chunks_by_agent[aid]) >= 2, f"agent {aid} received too few chunks"
        assert all(c.agent_id == aid for c in chunks_by_agent[aid])


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


@pytest.mark.timeout(900)
def test_full_pipeline_with_checkpoint_pool(tmp_path):
    """Checkpoints are saved every N train steps, plus a final one, and the FIFO pool is respected."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 5000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        # ~300 train steps before the env-step budget stops the run: checkpoints at
        # 40, 80, ... and a final one at stop -> the FIFO pool keeps the last 5
        # (exact versions depend on timing).
        self_play={"checkpoint_interval": 40, "pool_size": 5},
    )
    Launcher(config).launch()
    ckpts = CheckpointManager(str(tmp_path / "checkpoints"), pool_size=5).list_checkpoints("agent_0")
    versions = [c.policy_version for c in ckpts]
    assert len(versions) == 5, versions
    assert all(a < b for a, b in zip(versions, versions[1:])), versions
    assert all(v > 0 and v % 40 == 0 for v in versions[:-1]), versions
    assert ckpts[-1].meta["final"] is True, "the newest checkpoint is the learner's final snapshot"
    assert not any(c.meta["final"] for c in ckpts[:-1])


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
    assert all(manager.list_checkpoints(aid) for aid in config.get_trainable_agent_ids())


class _MetricsRecorder:
    """Stand-in for ``WandBLogger``: records every learner metrics dict the monitor forwards."""

    records: list[tuple[str, int, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def log_config(self, config: dict) -> None:
        pass

    def log_train_step(self, agent_id: str, metrics: dict, step: int) -> None:
        self.records.append((agent_id, step, dict(metrics)))

    def finish(self) -> None:
        pass


@pytest.mark.timeout(900)
def test_multi_agent_learners_both_train_until_the_budget(tmp_path, monkeypatch):
    """Two learners share the workers; both keep training until the global budget stops the run.

    Regression (T2.5): with per-learner step budgets, the first learner to finish
    left the workers blocked on its full chunk queue and the other agent starved.
    """
    import colosseum.launcher as launcher_mod

    _MetricsRecorder.records = []
    monkeypatch.setattr(launcher_mod, "WandBLogger", _MetricsRecorder)
    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16, "device": "cpu"},
        metrics={"log_interval": 1},
    )
    launcher = launcher_mod.Launcher(config)
    launcher.launch()

    assert launcher.env_steps_done >= 3000
    agent_ids = config.get_trainable_agent_ids()
    assert len(agent_ids) == 2
    last_step = {aid: max((s for a, s, _ in _MetricsRecorder.records if a == aid), default=0)
                 for aid in agent_ids}
    max_progress = {aid: max((m["progress"] for a, _, m in _MetricsRecorder.records if a == aid),
                             default=0.0) for aid in agent_ids}
    # Both learners were still training in the second half of the budget ...
    assert all(p >= 0.5 for p in max_progress.values()), max_progress
    # ... and neither starved: they trained comparable numbers of steps.
    assert min(last_step.values()) >= 0.5 * max(last_step.values()) > 0, last_step


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


@pytest.mark.timeout(900)
def test_full_pipeline_lstm_core(tmp_path):
    """Recurrent end-to-end run (R1-01): worker chunks with LSTM state train in the learner."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    data = config.model_dump()
    data["networks"]["core"] = {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}}
    Launcher(ColosseumConfig(**data)).launch()
    assert CheckpointManager(str(tmp_path / "checkpoints")).list_checkpoints("agent_0")
