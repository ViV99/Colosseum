"""SP2 end-to-end runs through real worker / learner processes (spawn start method, T7.1); since SP5 (T2.5 b) on
the hub and its local node (``train``'s path), and the worker tests on a node's spawned worker with its learners
(against ``node_kit.FakeHub``)."""

from __future__ import annotations

import json

import pytest

from cli_runner import REPO_ROOT
from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import ColosseumConfig, load_config
from colosseum.core.types import Lineup, SeatAssignment
from game_helpers import make_test_run_dir
from hub_kit import make_hub
from node_kit import intake_drops, train_on_node, train_steps, wait_for, worker_counters


def example_config(name: str):
    return REPO_ROOT / "configs" / "examples" / name


def _config(name: str, tmp_path, **sections: dict) -> ColosseumConfig:
    """Example config with WandB off and section overrides (runs go to tmp_path via make_test_run_dir)."""
    data = load_config(example_config(name)).model_dump()
    data["metrics"]["use_wandb"] = False
    for section, values in sections.items():
        data[section].update(values)
    return ColosseumConfig(**data)


def test_worker_produces_chunks(tmp_path):
    """A spawned worker process sends full-length chunks for its agent: the node's worker feeds its agent's learner,
    which takes every chunk (a chunk of another length or agent is dropped by reason, A17) and trains on them."""
    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 1, "device": "cpu"},
    )
    agent_id = "agent_0"
    hub, node = train_on_node(tmp_path, config, learners=[agent_id], workers=1,
                              lineup=lambda w, e: Lineup("2p", [SeatAssignment(agent_id), SeatAssignment(agent_id)]))
    try:
        wait_for(lambda: worker_counters(hub).get("chunks_sent", 0) >= 5 and train_steps(hub, agent_id) >= 5,
                 timeout=120)
        wait_for(lambda: any(m.intake for m in hub.messages("metrics")), timeout=30)
    finally:
        node.stop()
        hub.stop()
    assert not intake_drops(hub, agent_id)                              # every chunk was full-length and its own


def test_worker_multi_agent_routing(tmp_path):
    """Chunks are routed to the learner of the agent that occupies the slot."""
    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 1, "device": "cpu"},
    )
    agent_ids = config.get_trainable_agent_ids()
    hub, node = train_on_node(tmp_path, config, learners=agent_ids, workers=1,
                              lineup=lambda w, e: Lineup("2p", [SeatAssignment(agent_ids[0]),
                                                                SeatAssignment(agent_ids[1])]))
    try:
        wait_for(lambda: min(train_steps(hub, aid) for aid in agent_ids) >= 2, timeout=120)
        wait_for(lambda: all(any(i.agent_id == aid for m in hub.messages("metrics") for i in m.intake)
                             for aid in agent_ids), timeout=30)
    finally:
        node.stop()
        hub.stop()
    for aid in agent_ids:
        assert train_steps(hub, aid) >= 2, f"agent {aid} received too few chunks"
        assert not intake_drops(hub, aid)                               # nothing misrouted (dropped by reason)


@pytest.mark.timeout(900)
def test_full_pipeline(tmp_path):
    """Single-agent self-play training runs to completion and saves checkpoints."""
    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    run = make_test_run_dir(config, tmp_path)
    make_hub(config, run).launch()
    assert CheckpointManager(run.checkpoints).list_checkpoints("agent_0")


@pytest.mark.timeout(900)
def test_full_pipeline_with_checkpoint_pool(tmp_path):
    """Checkpoints are saved every N train steps, plus a final one, and the FIFO pool is respected."""
    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 5000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        # ~300 train steps before the env-step budget stops the run: checkpoints at
        # 40, 80, ... and a final one at stop -> the FIFO pool keeps the last 5
        # (exact versions depend on timing).
        checkpoint={"interval": 40, "keep_last": 5, "keep_every": 0},
    )
    run = make_test_run_dir(config, tmp_path)
    make_hub(config, run).launch()
    ckpts = CheckpointManager(run.checkpoints, keep_last=5).list_checkpoints("agent_0")
    versions = [c.policy_version for c in ckpts]
    assert len(versions) == 5, versions
    assert all(a < b for a, b in zip(versions, versions[1:])), versions
    assert all(v > 0 and v % 40 == 0 for v in versions[:-1]), versions
    assert ckpts[-1].meta["final"] is True, "the newest checkpoint is the learner's final snapshot"
    assert not any(c.meta["final"] for c in ckpts[:-1])


@pytest.mark.timeout(900)
def test_multi_agent_pipeline(tmp_path):
    """Two-agent league training runs to completion."""
    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        checkpoint={"interval": 50},
    )
    run = make_test_run_dir(config, tmp_path)
    make_hub(config, run).launch()
    manager = CheckpointManager(run.checkpoints)
    assert all(manager.list_checkpoints(aid) for aid in config.get_trainable_agent_ids())


@pytest.mark.timeout(900)
def test_multi_agent_learners_both_train_until_the_budget(tmp_path):
    """Two learners share the workers; both keep training until the global budget stops the run.

    Regression (T2.5): with per-learner step budgets, the first learner to finish
    left the workers blocked on its full chunk queue and the other agent starved.
    Learner metrics are read from the run's ``metrics.jsonl`` (T6.3).
    """
    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16, "device": "cpu"},
        metrics={"log_interval": 1},
        # the learners' LR progress follows the hub's env-step counter, which moves with the node's heartbeats and
        # reaches them with the hub's Announce (spec 4.4): quarter-second beats for this three-second run
        distributed={"heartbeat_sec": 0.25},
    )
    run = make_test_run_dir(config, tmp_path)
    hub = make_hub(config, run)
    hub.launch()

    records = [json.loads(line) for line in run.metrics_path.read_text().splitlines()]
    train = [(r["agent"], r["train_step"], r) for r in records if r["kind"] == "train"]
    assert hub.env_steps_done >= 3000
    agent_ids = config.get_trainable_agent_ids()
    assert len(agent_ids) == 2
    last_step = {aid: max((s for a, s, _ in train if a == aid), default=0) for aid in agent_ids}
    max_progress = {aid: max((m["progress"] for a, _, m in train if a == aid), default=0.0)
                    for aid in agent_ids}
    # Both learners were still training in the second half of the budget ...
    assert all(p >= 0.5 for p in max_progress.values()), max_progress
    # ... and neither starved: they trained comparable numbers of steps.
    assert min(last_step.values()) >= 0.5 * max(last_step.values()) > 0, last_step


@pytest.mark.timeout(900)
def test_subprocess_vec_env_pipeline(tmp_path):
    """Nested spawn (worker -> env subprocesses) trains to completion."""
    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 2000},
        rollout={
            "num_workers": 1, "envs_per_worker": 4, "chunk_length": 8,
            "vec_env": "subprocess", "subproc_workers": 2, "match_refresh_interval_sec": 1.0,
        },
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    run = make_test_run_dir(config, tmp_path)
    make_hub(config, run).launch()
    assert CheckpointManager(run.checkpoints).list_checkpoints("agent_0")
    # Env children log to worker-<i>-env<global offset>.log in the run dir (T6.2).
    assert "worker-0-env0 started (pid" in (run.logs / "worker-0-env0.log").read_text()
    assert "worker-0-env2 started (pid" in (run.logs / "worker-0-env2.log").read_text()


@pytest.mark.timeout(900)
def test_full_pipeline_lstm_core(tmp_path):
    """Recurrent end-to-end run (R1-01): worker chunks with LSTM state train in the learner."""
    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    data = config.model_dump()
    data["networks"]["core"] = {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}}
    lstm_config = ColosseumConfig(**data)
    run = make_test_run_dir(lstm_config, tmp_path)
    make_hub(lstm_config, run).launch()
    assert CheckpointManager(run.checkpoints).list_checkpoints("agent_0")
