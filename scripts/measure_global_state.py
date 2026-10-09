"""Cost of ``global_state`` on team_tag (SP2 spec, risk "Размер global_state"; docs/benchmarks.md).

For ``configs/examples/team_tag.yaml`` as is (global_state on, critic encoder on) and with
``with_global_state: false`` and no critic encoder, the script:
1. collects chunks in-process with the real ``RolloutLoop`` (8 envs, the config's chunk length);
2. reports the numpy payload bytes per chunk: observation leaves, ``global_state`` leaves, total;
3. times ``APPO.train_step`` on ``learner.batch_chunks`` of those chunks (median of 20 steps after
   3 warm-up steps, 1 torch thread).

Usage::

    .venv/bin/python scripts/measure_global_state.py [--chunks 64] [--json out.json]
"""
from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG = REPO_ROOT / "configs" / "examples" / "team_tag.yaml"
AGENT = "agent_0"


def _nbytes(tree) -> int:
    from colosseum.core.tree import tree_leaves

    return 0 if tree is None else sum(int(getattr(leaf, "nbytes", 0)) for leaf in tree_leaves(tree))


def measure(config, num_chunks: int) -> dict:
    import torch

    from colosseum.algorithms.appo import APPO
    from colosseum.core.registry import build_model, env_spec, make_env
    from colosseum.core.roles import agent_role_spec, resolve_agent_roles
    from colosseum.core.specs import ActionSpec
    from colosseum.core.types import Lineup, SeatAssignment, TrajectoryChunk
    from colosseum.worker.rollout_loop import LoopIO, RolloutLoop

    torch.set_num_threads(1)
    spec = env_spec(config)
    roles = resolve_agent_roles(config, spec)
    role = agent_role_spec(spec, roles[AGENT])
    agent_config = config.get_agent_config(AGENT)
    factory = functools.partial(build_model, agent_config, role)
    payloads: list[dict] = []
    io = LoopIO(send_chunk=lambda c: payloads.append(c.to_payload()), poll_weights=lambda agent_id: None)
    lineups = [Lineup("2v2", [SeatAssignment(AGENT) for _ in range(4)]) for _ in range(8)]
    loop = RolloutLoop(worker_id=0, env_fn=functools.partial(make_env, config), num_envs=8,
                       chunk_length=config.rollout.chunk_length, agent_ids=[AGENT], agent_roles=roles,
                       model_factories={AGENT: factory}, io=io, lineups=lineups, weight_sync_interval=1e9, seed=0)
    try:
        while len(payloads) < num_chunks:
            loop.step()
    finally:
        loop.close()
    obs_bytes = statistics.mean(_nbytes(p["obs"]) for p in payloads)
    gs_bytes = statistics.mean(_nbytes(p["global_state"]) for p in payloads)
    total_bytes = statistics.mean(_nbytes(p) for p in payloads)
    algo = APPO(factory(), agent_config.algorithm, ActionSpec.from_space(role.action_space), device="cpu")
    batch_size = config.learner.batch_chunks
    times = []
    for i in range(23):
        batch = [TrajectoryChunk.from_payload(p) for p in payloads[(i * batch_size) % len(payloads):][:batch_size]]
        if len(batch) < batch_size:
            batch = [TrajectoryChunk.from_payload(p) for p in payloads[:batch_size]]
        start = time.perf_counter()
        algo.train_step(batch)
        if i >= 3:
            times.append(time.perf_counter() - start)
    return {"chunk_length": config.rollout.chunk_length, "obs_bytes_per_chunk": obs_bytes,
            "global_state_bytes_per_chunk": gs_bytes, "payload_bytes_per_chunk": total_bytes,
            "batch_chunks": batch_size, "train_step_ms_median": 1000 * statistics.median(times)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from colosseum.core.config import load_config

    base = load_config(CONFIG)
    kwargs = {**base.env.kwargs, "with_global_state": False}
    without = load_config(CONFIG, {"env.kwargs": kwargs, "networks.critic_encoder_class": None})
    results = {"with_global_state": measure(base, args.chunks), "without_global_state": measure(without, args.chunks)}
    for name, r in results.items():
        print(f"{name}: {json.dumps(r)}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
