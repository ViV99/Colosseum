"""BC on a scripted expert reaches high accuracy and plays well (T4.4)."""
import numpy as np
import torch

from colosseum.bc.offline_bc import OfflineBCTrainer
from colosseum.networks.model import act
from helpers import MaskedToyEnv, make_simple_model


def _record_expert(num_steps: int, seed: int):
    env = MaskedToyEnv()
    obs, info = env.reset(seed=seed)
    data = {"observations": [], "actions": [], "action_masks": [], "dones": []}
    for _ in range(num_steps):
        action = env.expert_action()
        data["observations"].append(obs[0])
        data["actions"].append(action)
        data["action_masks"].append(info[0]["action_mask"])
        obs, reward, terminated, truncated, info = env.step({0: action})
        assert reward[0] == 1.0
        done = terminated[0] or truncated[0]
        data["dones"].append(done)
        if done:
            obs, info = env.reset()
    return {
        "observations": torch.as_tensor(np.stack(data["observations"])),
        "actions": torch.as_tensor(data["actions"], dtype=torch.long),
        "action_masks": torch.as_tensor(np.stack(data["action_masks"])),
        "dones": torch.as_tensor(data["dones"]),
    }


def test_bc_imitates_masked_scripted_expert():
    data = _record_expert(2000, seed=0)
    model = make_simple_model(obs_dim=4)
    trainer = OfflineBCTrainer(model, lr=1e-2)
    trainer.add_data(**data)
    metrics = trainer.train(num_epochs=30, batch_size=256)
    assert metrics["accuracy"] >= 0.95

    env = MaskedToyEnv()
    obs, info = env.reset(seed=123)
    rewards = []
    model.eval()
    for _ in range(500):
        mask = torch.as_tensor(info[0]["action_mask"])[None]
        with torch.no_grad():
            out = act(model, torch.as_tensor(obs[0])[None], model.initial_state(1), mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step({0: int(out.actions[0])})
        rewards.append(reward[0])
        if terminated[0] or truncated[0]:
            obs, info = env.reset()
    assert np.mean(rewards) >= 0.9
