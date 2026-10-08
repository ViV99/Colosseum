"""Per-process seed streams: local-mode and distributed learners (T6.1 fix round 1)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
import yaml

from colosseum.core.config import ColosseumConfig
from colosseum.utils.seeding import derive_seed, learner_seed

TTT = "examples.tic_tac_toe"


def cfg_data(seed=None, agents=None) -> dict:
    data = {
        "env": {"env_class": f"{TTT}.env.TicTacToeEnv", "num_players": 2},
        "networks": {
            "encoder_class": f"{TTT}.networks.TicTacToeEncoder",
            "policy_class": f"{TTT}.networks.TicTacToePolicy",
            "value_class": f"{TTT}.networks.TicTacToeValue",
        },
        "learner": {"device": "cpu"},
        "rollout": {"num_workers": 1, "envs_per_worker": 2},
        "training": {"seed": seed},
    }
    if agents:
        data["agents"] = {aid: None for aid in agents}
    return data


def _flat(state: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state.values()])


@pytest.fixture
def capture_learner_model(monkeypatch):
    """Replace learner_process with one that builds the algorithm and records its initial weights."""
    import colosseum.core.threads as threads_module
    import colosseum.learner.learner as learner_module

    captured: list[np.ndarray] = []

    def fake_learner_process(*, algorithm_factory, **kwargs):
        algo = algorithm_factory()
        captured.append(_flat(algo.model.state_dict()))

    monkeypatch.setattr(learner_module, "learner_process", fake_learner_process)
    monkeypatch.setattr(threads_module, "configure_torch_threads", lambda *a, **k: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return captured


def test_learner_seed_streams_are_deterministic_and_distinct():
    assert learner_seed(None, 0) is None
    assert learner_seed(7, 0) == learner_seed(7, 0) == derive_seed(7, "learner", 0)
    assert learner_seed(7, 0) != learner_seed(7, 1)
    assert learner_seed(7, 0) != learner_seed(8, 0)
    worker_streams = {7 + w * 1000 for w in range(64)}
    assert not {learner_seed(7, i) for i in range(16)} & worker_streams
    assert all(0 <= learner_seed(s, i) < 2**32 for s in (-5, 0, 2**40) for i in range(3))


def test_local_learner_weights_follow_training_seed(capture_learner_model, restore_global_rng):
    from colosseum.launcher import _learner_target

    cfg = ColosseumConfig.model_validate(cfg_data(seed=3))

    def run(seed):
        torch.manual_seed(12345 + len(capture_learner_model))  # a different global state each time
        _learner_target(
            agent_id="agent_0", config=cfg, trajectory_queue=None, weight_queues=[],
            stop_event=None, metrics_queue=None, seed=seed,
        )
        return capture_learner_model[-1]

    a, b, c = run(learner_seed(3, 0)), run(learner_seed(3, 0)), run(learner_seed(4, 0))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, run(learner_seed(3, 1)))


def test_launcher_gives_each_learner_its_own_seed_stream(monkeypatch, restore_global_rng):
    """Two agents in one run: each learner process gets learner_seed(seed, agent_index)."""
    import colosseum.launcher as launcher_module

    learner_kwargs: list[dict] = []

    class _Stop(Exception):
        pass

    class FakeProcess:
        def __init__(self, target, kwargs, daemon=None):
            self.target, self.kwargs = target, kwargs

        def start(self):
            if self.target is launcher_module._learner_target:
                learner_kwargs.append(self.kwargs)
            else:
                raise _Stop  # first worker: every learner has been started

    monkeypatch.setattr(launcher_module.mp, "Process", FakeProcess)
    cfg = ColosseumConfig.model_validate(cfg_data(seed=11, agents=["alpha", "beta"]))
    with pytest.raises(_Stop):
        launcher_module.Launcher(cfg).launch()
    assert [k["agent_id"] for k in learner_kwargs] == ["alpha", "beta"]
    assert [k["seed"] for k in learner_kwargs] == [learner_seed(11, 0), learner_seed(11, 1)]

    learner_kwargs.clear()
    unseeded = ColosseumConfig.model_validate(cfg_data(seed=None, agents=["alpha", "beta"]))
    with pytest.raises(_Stop):
        launcher_module.Launcher(unseeded).launch()
    assert [k["seed"] for k in learner_kwargs] == [None, None]


def test_run_learner_seeds_before_building_the_model(
    tmp_path, monkeypatch, capture_learner_model, restore_global_rng, restore_root_logging,
):
    import colosseum.distributed as distributed
    import colosseum.transport.grpc_transport as grpc_transport
    import colosseum.weight_store.grpc_store as grpc_store

    class FakeServer:
        def stop(self, grace):
            pass

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(grpc_transport, "serve_trajectory_receiver", lambda *a, **k: FakeServer())
    monkeypatch.setattr(grpc_store, "GRPCWeightStore", FakeStore)
    monkeypatch.setattr(distributed, "_install_stop_signal_handlers", lambda stop_event: None)

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg_data(seed=5, agents=["alpha", "beta"])))

    def run(agent_id, seed):
        torch.manual_seed(999 + len(capture_learner_model))
        distributed.run_distributed_learner(
            str(path), agent_id, 0, "localhost:1",
            overrides={"training.seed": seed, "checkpoint.dir": str(tmp_path / "ckpt")},
        )
        return capture_learner_model[-1]

    a, b = run("alpha", 5), run("alpha", 5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, run("alpha", 6))
    assert not np.array_equal(a, run("beta", 5))  # distinct per-agent stream

    # Same stream as a local-mode learner of the same agent.
    from colosseum.launcher import _learner_target

    cfg = ColosseumConfig.model_validate(cfg_data(seed=5, agents=["alpha", "beta"]))
    _learner_target(
        agent_id="alpha", config=cfg.get_agent_config("alpha"), trajectory_queue=None,
        weight_queues=[], stop_event=None, metrics_queue=None, seed=learner_seed(5, 0),
    )
    assert np.array_equal(a, capture_learner_model[-1])
