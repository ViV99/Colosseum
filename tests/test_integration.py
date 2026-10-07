"""Integration test for the full Colosseum pipeline."""

import multiprocessing as mp
import sys
import time

sys.path.insert(0, ".")


def test_worker_produces_chunks():
    """Test that a worker process produces trajectory chunks."""
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

    p = mp.Process(
        target=_worker_target,
        args=(0, config, [agent_id], agent_configs,
              trajectory_queues, weight_queues, stop_event, 500),
        daemon=True,
    )
    p.start()

    chunks_received = 0
    for i in range(5):
        try:
            chunk = trajectory_queues[agent_id].get(timeout=10)
            chunks_received += 1
            assert chunk.agent_id == agent_id
            assert chunk.observations.shape[0] == 8  # chunk_length
            print(f"  Chunk {i}: obs={chunk.observations.shape}, actions={chunk.actions.shape}")
        except Exception as e:
            print(f"  Timeout on chunk {i}: {e}")
            break

    stop_event.set()
    p.join(timeout=5)
    if p.is_alive():
        p.terminate()

    assert chunks_received > 0, "Worker did not produce any chunks!"
    print(f"  Worker produced {chunks_received} chunks. PASSED")


def test_learner_trains():
    """Test that a learner process trains from chunks."""
    from colosseum.algorithms.appo import APPO
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.core.registry import import_class
    from colosseum.core.types import TrajectoryChunk
    from colosseum.networks.actor_critic import ActorCriticNetwork

    config = load_config("configs/examples/tic_tac_toe.yaml")
    cd = config.model_dump()
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    # Create network and algorithm in this process
    encoder = import_class(config.networks.encoder_class)(**config.networks.kwargs)
    policy = import_class(config.networks.policy_class)(**config.networks.kwargs)
    value = import_class(config.networks.value_class)(**config.networks.kwargs)
    net = ActorCriticNetwork(encoder, policy, value)
    algo = APPO(net, config.algorithm, device="cpu")

    # Create synthetic chunks
    import torch

    chunks = []
    for _ in range(4):
        T = config.rollout.chunk_length
        obs = torch.randn(T, 3, 3, 3)
        actions = torch.randint(0, 9, (T,))
        log_probs = torch.randn(T)
        rewards = torch.randn(T)
        dones = torch.zeros(T)
        dones[-1] = 1.0
        values = torch.randn(T)
        bootstrap = torch.tensor(0.0)

        chunk = TrajectoryChunk(
            agent_id="agent_0",
            observations=obs,
            actions=actions,
            action_log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            bootstrap_value=bootstrap,
            behavior_policy_version=0,
        )
        chunks.append(chunk)

    # Train step
    metrics = algo.train_step(chunks)
    print(f"  Train step metrics: loss={metrics['total_loss']:.4f}, "
          f"policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}")
    assert "total_loss" in metrics
    assert algo.policy_version == 1
    print("  Learner train step PASSED")


def test_full_pipeline():
    """Test the full launcher pipeline end-to-end."""
    from colosseum.core.config import ColosseumConfig, load_config
    from colosseum.launcher import Launcher

    config = load_config("configs/examples/tic_tac_toe.yaml")
    cd = config.model_dump()
    cd["training"]["total_timesteps"] = 3000
    cd["rollout"]["num_workers"] = 1
    cd["rollout"]["envs_per_worker"] = 2
    cd["rollout"]["chunk_length"] = 8
    cd["learner"]["batch_chunks"] = 2
    cd["learner"]["queue_size"] = 16
    cd["metrics"]["use_wandb"] = False
    config = ColosseumConfig(**cd)

    launcher = Launcher(config)
    launcher.launch()
    print("  Full pipeline PASSED")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("=" * 60)
    print("Test 1: Learner trains from synthetic chunks")
    print("=" * 60)
    test_learner_trains()

    print()
    print("=" * 60)
    print("Test 2: Worker produces trajectory chunks")
    print("=" * 60)
    test_worker_produces_chunks()

    print()
    print("=" * 60)
    print("Test 3: Full pipeline end-to-end")
    print("=" * 60)
    test_full_pipeline()

    print()
    print("ALL TESTS PASSED!")
