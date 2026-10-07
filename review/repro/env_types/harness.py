"""Run the real rollout worker in-process and collect chunks/results."""
from __future__ import annotations

import queue
import threading
from functools import partial

import torch

from colosseum.networks.actor_critic import ActorCriticNetwork
from colosseum.worker.rollout_worker import rollout_worker_process


def make_net(enc, pol, val, recurrent=None):
    return ActorCriticNetwork(enc, pol, val, recurrent=recurrent)


def run_worker(env_fn, net_factory, steps=200, num_envs=2, chunk_length=8,
               slot_agent_map=None, agent_ids=("a",), seed=0, net_factories=None):
    agent_ids = list(agent_ids)
    tq = {a: queue.Queue(maxsize=100000) for a in agent_ids}
    wq = {a: queue.Queue() for a in agent_ids}
    rq = queue.Queue(maxsize=100000)
    stop = threading.Event()
    factories = net_factories or {a: net_factory for a in agent_ids}
    rollout_worker_process(
        worker_id=0, env_fn=env_fn, num_envs=num_envs, chunk_length=chunk_length,
        agent_ids=agent_ids, network_factories=factories,
        trajectory_queues=tq, weight_queues=wq, stop_event=stop,
        total_timesteps=steps, slot_agent_map=slot_agent_map,
        results_queue=rq, seed=seed,
    )
    chunks = {a: [] for a in agent_ids}
    for a in agent_ids:
        while not tq[a].empty():
            chunks[a].append(tq[a].get())
    results = []
    while not rq.empty():
        results.append(rq.get())
    return chunks, results
