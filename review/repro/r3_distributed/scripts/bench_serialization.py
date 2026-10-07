"""Measure chunk / weight serialization cost: torch.save+lz4 (current) vs raw numpy bytes."""
import io
import statistics
import time

import lz4.frame
import numpy as np
import torch

torch.set_num_threads(1)

from colosseum.core.types import TrajectoryChunk
from colosseum.transport import colosseum_pb2
from colosseum.transport.serialization import (
    deserialize_chunk, deserialize_state_dict, serialize_chunk, serialize_state_dict,
)


def make_chunk(T, obs_shape, n_act, masks=False, random_obs=True):
    obs = torch.randn(T, *obs_shape) if random_obs else (torch.rand(T, *obs_shape) > 0.8).float()
    return TrajectoryChunk(
        agent_id="a",
        observations=obs,
        actions=torch.randint(0, n_act, (T,)),
        action_log_probs=torch.randn(T),
        rewards=torch.randn(T),
        dones=torch.zeros(T),
        values=torch.randn(T),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=3,
        action_masks=torch.ones(T, n_act, dtype=torch.bool) if masks else None,
    )


FIELDS = ["observations", "actions", "action_log_probs", "rewards", "dones", "values", "bootstrap_value"]


def raw_ser(chunk, compress):
    parts = []
    for f in FIELDS:
        a = getattr(chunk, f).numpy()
        parts.append(np.ascontiguousarray(a).tobytes())
    data = b"".join(parts)
    if compress:
        data = lz4.frame.compress(data, compression_level=0)
    return data


def raw_deser(data, compress, T, obs_shape):
    if compress:
        data = lz4.frame.decompress(data)
    off = 0
    out = {}
    specs = [("observations", np.float32, (T, *obs_shape)), ("actions", np.int64, (T,)),
             ("action_log_probs", np.float32, (T,)), ("rewards", np.float32, (T,)),
             ("dones", np.float32, (T,)), ("values", np.float32, (T,)), ("bootstrap_value", np.float32, ())]
    for name, dt, shp in specs:
        n = int(np.prod(shp)) * np.dtype(dt).itemsize
        out[name] = torch.from_numpy(np.frombuffer(data, dtype=dt, count=int(np.prod(shp)), offset=off).reshape(shp))
        off += n
    return out


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e6


def bench_chunk(name, T, obs_shape, n_act, random_obs=True, n=200):
    c = make_chunk(T, obs_shape, n_act, random_obs=random_obs)
    raw_bytes = sum(getattr(c, f).numel() * getattr(c, f).element_size() for f in FIELDS)
    data, comp = serialize_chunk(c)
    data_nc, _ = serialize_chunk(c, compress=False)
    ser_us = timeit(lambda: serialize_chunk(c), n)
    ser_nc_us = timeit(lambda: serialize_chunk(c, compress=False), n)
    de_us = timeit(lambda: deserialize_chunk("a", 3, data, comp), n)
    proto_us = timeit(lambda: colosseum_pb2.TrajectoryChunkProto(agent_id="a", behavior_policy_version=3,
                                                                tensor_data=data, compressed=True).SerializeToString(), n)
    r = raw_ser(c, True)
    r_nc = raw_ser(c, False)
    rser_us = timeit(lambda: raw_ser(c, False), n)
    rser_c_us = timeit(lambda: raw_ser(c, True), n)
    rde_us = timeit(lambda: raw_deser(r_nc, False, T, obs_shape), n)
    print(f"[{name}] T={T} obs={obs_shape} payload_raw={raw_bytes/1024:.1f}KiB")
    print(f"   current torch.save+lz4: {len(data)/1024:.1f}KiB  (no-lz4 {len(data_nc)/1024:.1f}KiB)  "
          f"ser={ser_us:.0f}us (no-lz4 {ser_nc_us:.0f}us)  deser={de_us:.0f}us  proto_wrap={proto_us:.0f}us")
    print(f"   raw numpy bytes:        {len(r_nc)/1024:.1f}KiB (lz4 {len(r)/1024:.1f}KiB)  "
          f"ser={rser_us:.0f}us (lz4 {rser_c_us:.0f}us)  deser={rde_us:.0f}us")


def bench_weights(name, n_params_layers, n=30):
    net = torch.nn.Sequential(*[torch.nn.Linear(a, b) for a, b in n_params_layers])
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    nparams = sum(v.numel() for v in sd.values())
    data, comp = serialize_state_dict(sd)
    ser_us = timeit(lambda: serialize_state_dict(sd), n)
    de_us = timeit(lambda: deserialize_state_dict(data, comp), n)
    # server-side GetWeights path: InMemoryWeightStore.get (torch.load) + serialize_state_dict again
    from colosseum.weight_store.shared_memory import InMemoryWeightStore
    from colosseum.core.types import WeightPayload
    s = InMemoryWeightStore()
    s.put("a", WeightPayload("a", 1, sd))
    get_us = timeit(lambda: serialize_state_dict(s.get("a").state_dict), n)
    print(f"[weights {name}] params={nparams/1e6:.2f}M raw={nparams*4/2**20:.1f}MiB  "
          f"wire(lz4)={len(data)/2**20:.2f}MiB  ratio={len(data)/(nparams*4):.3f}  "
          f"put-ser={ser_us/1000:.1f}ms  client-deser={de_us/1000:.1f}ms  "
          f"server-GetWeights-reserialize={get_us/1000:.1f}ms per request")


if __name__ == "__main__":
    bench_chunk("tictactoe", 32, (3, 3, 3), 9, random_obs=False)
    bench_chunk("claude.md typical", 256, (256,), 16)
    bench_chunk("claude.md typical sparse-binary obs", 256, (256,), 16, random_obs=False)
    bench_chunk("image-ish", 128, (8, 32, 32), 16, random_obs=False, n=50)
    bench_weights("small 0.2M", [(256, 256), (256, 256), (256, 256)])
    bench_weights("medium 4.2M", [(1024, 1024)] * 4)
    bench_weights("large 33.6M", [(2048, 2048)] * 8, n=5)
