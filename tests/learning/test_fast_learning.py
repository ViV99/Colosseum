"""Bandit and short chain are solved in seconds by the real RolloutLoop + APPO (spec §3.3)."""
from __future__ import annotations

import functools
import time

import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk, WeightPayload
from colosseum.networks.model import act
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from helpers import make_simple_model
from learning_envs import ContextualBandit, ShortChain

pytestmark = pytest.mark.usefixtures("restore_global_rng")


def train_until_solved(env_cls, obs_dim, solved, *, gamma=0.9, lr=3e-3, num_envs=8, chunk_length=16,
                       batch_chunks=4, max_updates=400, seed=0) -> int:
    """Run collection and training in-process. Returns the number of updates used, or -1."""
    torch.manual_seed(seed)
    model_fn = functools.partial(make_simple_model, obs_dim=obs_dim, hidden_dim=32, num_actions=2)
    algo = APPO(model_fn(), AlgorithmConfig(learning_rate=lr, lr_schedule="constant", gamma=gamma,
                                            entropy_coeff=0.01), device="cpu")
    chunks: list[TrajectoryChunk] = []
    latest = {"payload": WeightPayload.from_model("agent_0", algo.policy_version, algo.model)}
    io = LoopIO(
        send_chunk=lambda c: chunks.append(TrajectoryChunk.from_payload(c.to_payload())),
        poll_weights=lambda agent_id: latest["payload"],
    )
    loop = RolloutLoop(worker_id=0, env_fn=env_cls, num_envs=num_envs, chunk_length=chunk_length,
                       agent_ids=["agent_0"], model_factories={"agent_0": model_fn}, io=io,
                       gamma=gamma, weight_sync_interval=0.0, seed=seed)
    try:
        for update in range(1, max_updates + 1):
            while len(chunks) < batch_chunks:
                loop.step()
            batch = chunks[:batch_chunks]
            del chunks[:batch_chunks]
            algo.set_progress(update / max_updates)
            algo.train_step(batch)
            latest["payload"] = WeightPayload.from_model("agent_0", algo.policy_version, algo.model)
            loop.sync_weights()
            if update % 10 == 0 and solved(algo.model):
                return update
    finally:
        loop.close()
    return -1


@torch.no_grad()
def greedy_and_confident(model, obs: torch.Tensor, best: torch.Tensor, min_prob: float) -> bool:
    state = model.initial_state(obs.shape[0])
    out = act(model, obs, state, deterministic=True)
    if not torch.equal(out.actions.reshape(-1).long(), best):
        return False
    probs = model.step(obs, state).dist.log_prob(best).exp()
    return bool((probs >= min_prob).all())


def test_contextual_bandit_is_solved_in_seconds():
    start = time.monotonic()
    updates = train_until_solved(
        ContextualBandit, 2,
        lambda m: greedy_and_confident(m, torch.eye(2), torch.tensor([0, 1]), min_prob=0.9))
    assert updates != -1, "bandit not solved within 400 updates"
    assert time.monotonic() - start < 60


def test_short_chain_is_solved_in_seconds():
    start = time.monotonic()
    states = torch.eye(ShortChain.LENGTH)[: ShortChain.LENGTH - 1]  # every non-terminal state
    updates = train_until_solved(
        ShortChain, ShortChain.LENGTH,
        lambda m: greedy_and_confident(m, states, torch.ones(ShortChain.LENGTH - 1, dtype=torch.long), 0.8))
    assert updates != -1, "chain not solved within 400 updates"
    assert time.monotonic() - start < 60
