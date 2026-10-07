"""Throughput of one rollout worker (in-process, chunks discarded into a list).
usage: bench.py <tic_tac_toe|space_miners> <num_envs> <steps> <threads|default> [subprocess] [profile]"""
import sys, time, cProfile, pstats, io
from common import *
from functools import partial
from colosseum.core.config import load_config
from colosseum.launcher import _create_env, _create_network
def main():
    name, n_envs, steps, thr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    kind = "subprocess" if "subprocess" in sys.argv else "sync"
    if thr != "default": torch.set_num_threads(int(thr))
    cfg = load_config(f"/home/viv/dev/repos/Colosseum/configs/examples/{name}.yaml")
    env_fn = partial(_create_env, cfg.env.env_class, cfg.env.kwargs)
    tq = {"a": ListQ()}
    args = dict(worker_id=0, env_fn=env_fn, num_envs=n_envs, chunk_length=cfg.rollout.chunk_length, agent_ids=["a"],
                network_factories={"a": partial(_create_network, cfg)}, trajectory_queues=tq, weight_queues={"a": ListQ()},
                stop_event=threading.Event(), total_timesteps=steps, results_queue=ListQ(), seed=0, vec_env_kind=kind, subproc_workers=2)
    prof = "profile" in sys.argv
    t0 = time.perf_counter()
    if prof:
        pr = cProfile.Profile(); pr.enable()
    rollout_worker_process(**args)
    dt = time.perf_counter() - t0
    print(f"{name} envs={n_envs} threads={torch.get_num_threads()} vec={kind}: {steps/dt:.0f} env-steps/s, {2*steps/dt:.0f} agent-steps/s, chunks={len(tq['a'].items)}")
    if prof:
        pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18); print(s.getvalue()[:5000])

if __name__ == '__main__':
    main()
