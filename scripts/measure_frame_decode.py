"""SP5 criterion 5 (spec §3): parsing a 256-slot chunk of space_miners' size (~260 KB) together with the hostile-input
checks and the chunk checks is at least 10x faster on the new frame than on today's npz + JSON path with
``validate_chunk_payload``.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_frame_decode.py [--repeats 1000] [--json OUT]

Both sides start from the bytes a learner receives and end with a checked numpy payload:
- old (today's wire, ``GRPCTransport.send_chunk``): ``deserialize_chunk_payload`` of the lz4 npz + JSON blob +
  ``validate_chunk_payload`` (field checks, building the chunk once, slot rules); also reported without lz4;
- new (a worker's frame to a learner on its machine, never compressed): ``decode_frame`` (hostile-input checks) +
  ``chunk_meta`` + ``check_payload_fields`` + ``ChunkValidator.check`` (role checks, slot rules, finiteness, zero
  rewards, action ranges); also reported for an lz4 frame (another machine).
It also reports, for information (amendment A13; not part of the rule), the full path to a built
``TrajectoryChunk`` on both sides (old: unpack + ``validate_chunk_payload`` + ``from_payload``; new:
``decode_chunk(..., validator=...)`` = frame decode + checks + ``from_payload``): building torch tensors is the same
work on both sides and not parsing.

The role is space_miners' (``examples/space_miners/game.py``; spaces copied here so Box2D is not needed): Dict
observations of 924 B per slot, ``Units(3, Dict(accel=Box(2), push=Discrete(2)))`` actions with masks, 3 deciders.
The chunk is built by the worker's own ``RolloutBuffer`` (255 ACTs and a BOOT) with random content.

Rule (fixed before measuring, T1.1): PASS iff median(old, lz4) / median(new, plain) >= 10. Medians of ``--repeats``
timed runs after 100 warm-up ones, ``time.perf_counter`` per run (a duration on one machine, not a wall-clock
comparison); alone on a quiet machine, two runs, both must pass.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import gymnasium
import numpy as np

from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.net.chunks import ChunkValidator, check_payload_fields, chunk_meta, decode_chunk, encode_chunk
from colosseum.net.frame import decode_frame, payload_byte_cap
from colosseum.transport.serialization import (
    deserialize_chunk_payload,
    serialize_chunk_payload,
    validate_chunk_payload,
)
from colosseum.worker.buffers import BufferSpec, RolloutBuffer

SLOTS = 256
RATIO_MIN = 10.0
WARMUP = 100
MAX_SHIPS, SHIP_FEATURES, ASTEROID_FEATURES, MAX_ASTEROIDS = 3, 9, 7, 24   # examples/space_miners/game.py


def space_miners_role() -> RoleSpec:
    box = gymnasium.spaces.Box
    obs = gymnasium.spaces.Dict([
        ("ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
        ("enemy_ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
        ("asteroids", box(-1.0, 1.0, (MAX_ASTEROIDS, ASTEROID_FEATURES), np.float32)),
        ("asteroid_mask", gymnasium.spaces.MultiBinary(MAX_ASTEROIDS)),
        ("global", box(-1.0, 1.0, (3,), np.float32)),
    ])
    per_ship = gymnasium.spaces.Dict([("accel", box(-1.0, 1.0, (2,), np.float32)),
                                      ("push", gymnasium.spaces.Discrete(2))])
    return RoleSpec(obs, Units(MAX_SHIPS, per_ship))


def build_chunk(role: RoleSpec, seed: int = 0):
    rng = np.random.default_rng(seed)
    spec = BufferSpec(ObsSpec.from_space(role.observation_space), ActionSpec.from_space(role.action_space), None)
    buf = RolloutBuffer(SLOTS, spec)
    buf.begin(None, 7)
    mask = spec.action.full_mask(())
    for _ in range(SLOTS - 1):
        obs = {k: (rng.random(v.shape) * 2 - 1).astype(v.dtype) if v.dtype == np.float32
               else rng.integers(0, 2, v.shape).astype(v.dtype) for k, v in spec.obs.allocate(()).items()}
        action = {"accel": rng.uniform(-1, 1, (MAX_SHIPS, 2)).astype(np.float32),
                  "push": rng.integers(0, 2, MAX_SHIPS).astype(np.int64)}
        buf.write_act(obs, None, mask, action, -1.5, np.full(MAX_SHIPS, -0.5, np.float32), 0.01)
    buf.write_boot(obs, None, reset_after=False)
    return buf.build_chunk("main")


def timed(fn, repeats: int) -> float:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--json", type=str, default=None, help="write the numbers to this file")
    args = parser.parse_args()
    role = space_miners_role()
    chunk = build_chunk(role)
    validator = ChunkValidator(role, agent_id="main", teacher=False, chunk_length=SLOTS)
    cap = payload_byte_cap(64)
    payload = chunk.to_payload()
    blobs = {"old_lz4": serialize_chunk_payload(payload, compress=True)[0],
             "old_plain": serialize_chunk_payload(payload, compress=False)[0],
             "new_plain": encode_chunk(chunk, generation=1, node=1, worker=0),
             "new_lz4": encode_chunk(chunk, generation=1, node=1, worker=0, compress=True)}

    def old(data: bytes, compressed: bool, build: bool = False) -> None:
        decoded = deserialize_chunk_payload(data, compressed)
        validate_chunk_payload(decoded)
        if build:
            TrajectoryChunk.from_payload(decoded)

    def new(data: bytes) -> None:
        decoded, meta = decode_frame(data, cap=cap, kind="chunk")
        info = chunk_meta(meta)
        check_payload_fields(decoded)
        validator.check(decoded, info.agent_id, info.policy_version)

    t = {"old_lz4": timed(lambda: old(blobs["old_lz4"], True), args.repeats),
         "old_plain": timed(lambda: old(blobs["old_plain"], False), args.repeats),
         "new_plain": timed(lambda: new(blobs["new_plain"]), args.repeats),
         "new_lz4": timed(lambda: new(blobs["new_lz4"]), args.repeats),
         "old_lz4_chunk": timed(lambda: old(blobs["old_lz4"], True, build=True), args.repeats),
         "new_plain_chunk": timed(lambda: decode_chunk(blobs["new_plain"], cap=cap, validator=validator),
                                  args.repeats)}
    ratio = t["old_lz4"] / t["new_plain"]
    full_ratio = t["old_lz4_chunk"] / t["new_plain_chunk"]
    verdict = "PASS" if ratio >= RATIO_MIN else "FAIL"
    print(f"rule (W01-1, fixed before measuring): PASS iff median(old_lz4) / median(new_plain) >= {RATIO_MIN}, "
          f"from the received bytes to a checked numpy payload")
    for key, value in t.items():
        size = len(blobs[key.removesuffix("_chunk")])
        print(f"{key:16s} {size / 1024:7.1f} KB  median {value * 1e6:8.1f} us")
    print(f"criterion 5: old_lz4 / new_plain = {ratio:.1f} (>= {RATIO_MIN}: {verdict}); for information: "
          f"old_plain / new_plain = {t['old_plain'] / t['new_plain']:.1f}, old_lz4 / new_lz4 = "
          f"{t['old_lz4'] / t['new_lz4']:.1f}")
    print(f"full path to a built TrajectoryChunk (for information, A13): old_lz4_chunk (unpack + "
          f"validate_chunk_payload + from_payload) / new_plain_chunk (frame decode + checks + from_payload) = "
          f"{full_ratio:.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"slots": SLOTS, "repeats": args.repeats, "seconds": t,
                                               "bytes": {k: len(v) for k, v in blobs.items()}, "ratio": ratio,
                                               "full_path_ratio": full_ratio, "ratio_min": RATIO_MIN,
                                               "verdict": verdict}, indent=2) + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
