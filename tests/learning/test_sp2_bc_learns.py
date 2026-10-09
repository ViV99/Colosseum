"""BC on a masked scripted expert reaches high accuracy and plays well (SP1 guarantee on SP2, T6.3)."""
from __future__ import annotations

import numpy as np
import torch

from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.networks.model import act
from game_learning_envs import MaskedChoiceGame, make_mlp_model


def record_expert(num_decisions: int, seed: int) -> dict:
    env = MaskedChoiceGame()
    result = env.reset(seed, "solo")
    data = {"observations": [], "actions": [], "action_masks": [], "dones": []}
    for _ in range(num_decisions):
        action = env.expert_action()
        data["observations"].append(result.obs[0])
        data["actions"].append(action)
        data["action_masks"].append(result.action_masks[0])
        result = env.step({0: action})
        data["dones"].append(result.episode_over)
        if result.episode_over:
            result = env.reset(None, "solo")
    return {
        "observations": np.stack(data["observations"]),
        "actions": np.asarray(data["actions"], dtype=np.int64),
        "action_masks": np.stack(data["action_masks"]),
        "dones": np.asarray(data["dones"]),
    }


def test_bc_imitates_a_masked_scripted_expert(restore_global_rng):
    torch.manual_seed(0)
    spec = MaskedChoiceGame.spec.roles["player"]
    model = make_mlp_model(obs_dim=4, num_actions=4)
    trainer = OfflineBCTrainer(model, ActionSpec.from_space(spec.action_space),
                               ObsSpec.from_space(spec.observation_space), lr=1e-2)
    trainer.add_data(**record_expert(2000, seed=0))
    metrics = trainer.train(num_epochs=30, batch_size=256)
    assert metrics["accuracy"] >= 0.95

    env = MaskedChoiceGame()
    result = env.reset(123, "solo")
    rewards = []
    model.eval()
    for _ in range(500):
        obs = torch.as_tensor(result.obs[0])[None]
        mask = torch.as_tensor(result.action_masks[0])[None]
        out = act(model, obs, model.initial_state(1), mask, deterministic=True)
        result = env.step({0: int(out.actions.reshape(-1)[0])})
        rewards.append(result.rewards.get(0, 0.0))
        if result.episode_over:
            result = env.reset(None, "solo")
    assert np.mean(rewards) >= 0.9
