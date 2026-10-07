"""Targeted probes of the distributed layer behaviour."""
import multiprocessing as mp
import queue
import sys
import time

import torch

from colosseum.core.types import TrajectoryChunk, WeightPayload


def sd():
    return {"w": torch.randn(4, 4)}


def chunk(T=32):
    return TrajectoryChunk(
        agent_id="agent_0", observations=torch.randn(T, 27), actions=torch.randint(0, 9, (T,)),
        action_log_probs=torch.randn(T), rewards=torch.randn(T), dones=torch.zeros(T),
        values=torch.randn(T), bootstrap_value=torch.tensor(0.0), behavior_policy_version=1)


def probe_version_regression():
    """Learner restart (version restarts from 0) -> workers never take new weights."""
    from colosseum.distributed import GRPCWeightSink, GRPCWeightSource
    from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store
    srv = serve_weight_store(port=51110)
    store = GRPCWeightStore("localhost:51110")
    sink, src = GRPCWeightSink(store, "agent_0"), GRPCWeightSource(store, "agent_0")
    sink.put_nowait(WeightPayload("agent_0", 500, sd()))
    print("worker got v", src.get_nowait().policy_version)
    # learner restarts (no resume in run-learner) and publishes v0..v10
    for v in range(0, 11):
        sink.put_nowait(WeightPayload("agent_0", v, sd()))
    try:
        p = src.get_nowait()
        print("after restart worker got v", p.policy_version)
    except queue.Empty:
        print("after learner restart (store at v10) worker sees queue.Empty -> keeps stale v500 weights")
    store.close(); srv.stop(0)


def probe_weight_store_down():
    """Worker weight sync with weight store unreachable -> exception propagates?"""
    from colosseum.distributed import GRPCWeightSource
    from colosseum.weight_store.grpc_store import GRPCWeightStore
    store = GRPCWeightStore("localhost:51111")  # nothing listening
    src = GRPCWeightSource(store, "agent_0")
    try:
        src.get_nowait()
        print("no exception")
    except queue.Empty:
        print("Empty")
    except Exception as e:  # noqa
        print("get_nowait raised", type(e).__name__, getattr(e, "code", lambda: None)())


def probe_backpressure():
    """Learner not draining: how do sends behave?"""
    from colosseum.distributed import GRPCTrajectorySink
    from colosseum.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver
    q = queue.Queue(maxsize=4)
    srv = serve_trajectory_receiver(q, port=51112)
    t = GRPCTransport("localhost:51112")
    sink = GRPCTrajectorySink(t, "agent_0")
    c = chunk()
    for i in range(7):
        t0 = time.perf_counter()
        n = t.send_chunks_batch("agent_0", [c])
        print(f"send {i}: chunks_received={n} took {time.perf_counter()-t0:.2f}s qsize={q.qsize()}")
    t.close(); srv.stop(0)


def probe_send_latency():
    """Per-chunk RPC latency, stream-per-chunk vs one stream for many chunks (localhost)."""
    from colosseum.transport.grpc_transport import GRPCTransport, serve_trajectory_receiver
    q = queue.Queue(maxsize=100000)
    srv = serve_trajectory_receiver(q, port=51113)
    t = GRPCTransport("localhost:51113")
    for T, obs in [(32, 27), (256, 256)]:
        c = TrajectoryChunk(agent_id="agent_0", observations=torch.randn(T, obs), actions=torch.randint(0, 9, (T,)),
                            action_log_probs=torch.randn(T), rewards=torch.randn(T), dones=torch.zeros(T),
                            values=torch.randn(T), bootstrap_value=torch.tensor(0.0), behavior_policy_version=1)
        t.send_chunk("agent_0", c)
        N = 300
        t0 = time.perf_counter()
        for _ in range(N):
            t.send_chunk("agent_0", c)
        per = (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        t.send_chunks_batch("agent_0", [c] * N)
        per_b = (time.perf_counter() - t0) / N
        print(f"T={T} obs={obs}: send_chunk (stream per chunk) {per*1e3:.2f} ms/chunk -> {1/per:.0f} chunks/s; "
              f"one stream {per_b*1e3:.2f} ms/chunk -> {1/per_b:.0f} chunks/s")
        while not q.empty():
            q.get_nowait()
    t.close(); srv.stop(0)


def probe_local_weight_queue_staleness():
    """Local mode: learner put_nowait into maxsize=2 queue drops NEWEST payloads."""
    from colosseum.learner.learner import _push_weights
    from colosseum.worker.rollout_worker import _sync_weights

    class A:
        def __init__(self):
            self.network = torch.nn.Linear(2, 2); self.policy_version = 0
    a = A()
    q = mp.Queue(maxsize=2)
    net = torch.nn.Linear(2, 2)
    for v in range(1, 41):  # learner pushes v1..v40 between two worker syncs
        a.policy_version = v
        _push_weights(a, "agent_0", [q])
        time.sleep(0.005)
    got = _sync_weights(net, q)
    print(f"learner at v40, worker _sync_weights loaded v{got}")
    q.cancel_join_thread()


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"=== {name}")
        globals()[f"probe_{name}"]()
