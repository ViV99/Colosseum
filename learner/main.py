import os
import logging
import pathlib

import torch.cuda
import yaml
import torch.nn as nn

from learner.models import Snake3d
from learner.learner_instance import Learner
from learner.clients import init_arena_client
from learner.models.rl import Seq3dEncoder, ActorCritic, PPO

logger = logging.getLogger(__name__)

with open(pathlib.Path(__file__).parent.resolve() / "config" / "config.yaml") as f:
    config = yaml.safe_load(f)

init_arena_client(os.getenv("ARENA_URL", "http://arena-service:8000"))


if __name__ == "__main__":
    action_dim, hidden_dim = config["action_dim"], config["hidden_dim"]
    state_encoder = Seq3dEncoder(4, hidden_dim // 16)
    discrete_actor_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim // 2, action_dim),
        nn.Softmax(dim=-1)
    )
    critic_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim // 2),
        nn.Linear(hidden_dim // 2, 1),
    )
    ac = ActorCritic(
        state_encoder=state_encoder,
        critic_head=critic_head,
        discrete_actor_heads=nn.ModuleList([discrete_actor_head]),
        continuous_actor_heads=None,
        continuous_action_std_init=None,
    )
    ppo = PPO(
        policy=ac,
        num_epochs=10,
        lr_actor=5e-4,
        lr_critic=1e-3,
        gamma=0.99,
        eps_clip=0.2,
        max_seq_len=4
    )
    model = Snake3d(ppo)
    learner = Learner(
        inference_proc_count=4,
        batch_size=4,
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    learner.run()
