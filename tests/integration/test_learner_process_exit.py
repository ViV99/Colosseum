"""A learner process exits even when nobody reads its weight queue (T2.2).

Numpy weight payloads are pickled in full (unlike torch tensors, which travel as
small fd handles), so a payload larger than the pipe buffer keeps the queue's
feeder thread blocked until a worker reads it. Workers that already stopped
never do; the learner must not wait for them at exit.
"""

import multiprocessing as mp

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig, LearnerConfig
from colosseum.learner.learner import learner_process
from dataflow_helpers import TinyModel, chunk_payload

BIG_NUM_ACTIONS = 8192  # policy head 4 x 8192 floats: ~160 KB per weight payload


def _big_appo() -> APPO:
    return APPO(TinyModel(num_actions=BIG_NUM_ACTIONS), AlgorithmConfig(), device="cpu")


def test_learner_exits_with_unread_weight_payloads():
    ctx = mp.get_context("spawn")
    traj = ctx.Queue()
    unread_weights = ctx.Queue(maxsize=2)
    stop = ctx.Event()
    for version in range(2):
        traj.put(chunk_payload(T=4, version=version))
    proc = ctx.Process(target=learner_process, kwargs=dict(
        agent_id="a", algorithm_factory=_big_appo, trajectory_queue=traj,
        weight_queues=[unread_weights],
        config=LearnerConfig(batch_chunks=2, weight_push_interval=1, device="cpu"),
        stop_event=stop, total_train_steps=1,
    ))
    proc.start()
    try:
        proc.join(timeout=60)
        assert not proc.is_alive(), "learner hung at exit flushing an unread weight payload"
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()
        unread_weights.cancel_join_thread()
