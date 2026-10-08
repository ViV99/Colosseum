"""No torch.Tensor ever crosses a process boundary (spec block 2, R6-02).

Real producers (rollout_worker_process, learner_process, Launcher refresh) run
in-process with CheckedQueues that raise on any tensor.
"""
import threading
import time

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, ColosseumConfig, LearnerConfig, load_config
from colosseum.core.ipc import assert_no_tensors
from colosseum.core.types import TrajectoryChunk, WeightPayload, WorkerCommand, state_dict_to_numpy
from colosseum.learner.learner import learner_process
from colosseum.worker.rollout_worker import rollout_worker_process
from dataflow_helpers import (
    CheckedQueue,
    EnvFactory,
    GridStepEnv,
    TinyModel,
    chunk_payload,
    make_tiny_model,
)
from helpers import example_config


@pytest.fixture
def restore_torch_threads():
    n = torch.get_num_threads()
    yield
    torch.set_num_threads(n)


def test_worker_queues_carry_only_numpy(restore_torch_threads):
    tq, wq, rq, cq = CheckedQueue(), CheckedQueue(maxsize=1), CheckedQueue(), CheckedQueue()
    wq.put(WeightPayload.from_model("a", 3, TinyModel()))
    cq.put(WorkerCommand(
        slot_agent_map=[["a", "a"]], slot_network_map=[["latest", "ckpt_v1"]],
        collect_mask=[[True, False]],
        new_checkpoints={"a": {"ckpt_v1": state_dict_to_numpy(TinyModel().state_dict())}},
    ))
    rollout_worker_process(
        worker_id=0, env_fn=EnvFactory(GridStepEnv, lengths=(3,)), num_envs=1, chunk_length=4,
        agent_ids=["a"], model_factories={"a": make_tiny_model},
        trajectory_queues={"a": tq}, weight_queues={"a": wq}, stop_event=threading.Event(),
        weight_sync_interval=0.0, max_env_steps=24, results_queue=rq, command_queue=cq,
    )
    payloads = [tq.get_nowait() for _ in range(tq.qsize())]
    assert payloads and all(isinstance(p, dict) for p in payloads)
    chunks = [TrajectoryChunk.from_payload(p) for p in payloads]
    assert chunks[-1].behavior_policy_version == 3
    assert rq.qsize() > 0


def test_learner_queues_carry_only_numpy():
    traj, wq, mq, ckq = CheckedQueue(), CheckedQueue(maxsize=1), CheckedQueue(), CheckedQueue()
    for version in range(4):
        traj.put(chunk_payload(T=4, version=version))
    stop = threading.Event()
    thread = threading.Thread(target=learner_process, kwargs=dict(
        agent_id="a",
        algorithm_factory=lambda: APPO(TinyModel(), AlgorithmConfig(), device="cpu"),
        trajectory_queue=traj, weight_queues=[wq],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, metrics_queue=mq, checkpoint_queue=ckq, checkpoint_interval=1,
    ))
    thread.start()
    deadline = time.monotonic() + 60
    while ckq.qsize() < 2 and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    ckpt = ckq.get_nowait()
    assert all(isinstance(v, np.ndarray) for v in ckpt["state_dict"].values())
    assert isinstance(wq.get_nowait(), WeightPayload)
    assert mq.qsize() >= 1


def test_refresh_commands_carry_numpy_checkpoints(tmp_path):
    from colosseum.coordinator.coordinator import Coordinator
    from colosseum.core.registry import build_model
    from colosseum.launcher import Launcher

    data = load_config(example_config("tic_tac_toe.yaml")).model_dump()
    data["checkpoint"]["dir"] = str(tmp_path / "ckpt")
    data["training"]["phase"] = "self_play"
    data["self_play"]["latest_prob"] = 0.0
    data["rollout"]["envs_per_worker"] = 4
    data["metrics"]["use_wandb"] = False
    cfg = ColosseumConfig(**data)
    coord = Coordinator(cfg)
    coord.agent_pool.register_trainable("agent_0")
    coord.checkpoint_manager.save("agent_0", 50, build_model(cfg).state_dict())
    coord.setup_matchmaker("agent_0")
    cq = CheckedQueue()
    Launcher(cfg)._refresh_worker_matches(coord, ["agent_0"], [cq], [{"agent_0": set()}])
    cmd = cq.get_nowait()
    state_dict = cmd.new_checkpoints["agent_0"]["ckpt_v50"]
    assert all(isinstance(v, np.ndarray) for v in state_dict.values())


def test_learner_resumes_from_numpy_state_and_optimizer_tree():
    """resume_state crosses the process boundary as numpy (weights + optimizer tree)."""
    from colosseum.core.ipc import to_numpy_tree

    source = APPO(TinyModel(), AlgorithmConfig(), device="cpu")
    source.train_step([TrajectoryChunk.from_payload(chunk_payload(T=4, version=v)) for v in range(2)])
    resume_state = {
        "state_dict": state_dict_to_numpy(source.model.state_dict()),
        "optimizer_state": to_numpy_tree(source.optimizer_state_dict),
        "policy_version": 5,
    }
    assert_no_tensors(resume_state)
    built: list[APPO] = []

    def factory() -> APPO:
        built.append(APPO(TinyModel(), AlgorithmConfig(), device="cpu"))
        return built[-1]

    wq = CheckedQueue(maxsize=1)
    learner_process(
        agent_id="a", algorithm_factory=factory, trajectory_queue=CheckedQueue(), weight_queues=[wq],
        config=LearnerConfig(batch_chunks=2, device="cpu"), stop_event=threading.Event(),
        total_train_steps=5, resume_state=resume_state,  # budget already reached: resume, push, exit
    )
    restored = built[0]
    assert restored.policy_version == 5
    for key, value in source.model.state_dict().items():
        assert torch.equal(restored.model.state_dict()[key], value), key
    src_opt, got_opt = source.optimizer_state_dict, restored.optimizer_state_dict
    assert src_opt["state"].keys() == got_opt["state"].keys()
    for idx, slot in src_opt["state"].items():
        for name, value in slot.items():
            assert torch.equal(got_opt["state"][idx][name], value), (idx, name)
    pushed = wq.get_nowait()
    assert pushed.policy_version == 5
    for key, value in resume_state["state_dict"].items():
        assert np.array_equal(pushed.state_dict[key], value), key
