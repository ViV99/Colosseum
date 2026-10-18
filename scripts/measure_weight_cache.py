"""SP5 criterion 4, second half (spec §3): a worker reads a 50 MB model from the shared-memory weight cache (``mmap`` +
``np.frombuffer`` of every leaf, without ``load_state_dict``) while the learner publishes twice a second; the p95 of
the read must be at most 100 ms.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_weight_cache.py [--seconds 30] [--json OUT]

A writer process (spawn) publishes a 50 MB float32 state dict (100 leaves of 0.5 MB) with ``WeightCacheWriter.
publish_latest`` every 0.5 s, each version with new values. The reader (this process) loops for ``--seconds``:
``WeightCacheReader.read_latest`` of the newest version, then one read of every 4 KiB page of every leaf (so the
mapping is really read, not only set up). Each read is timed with ``time.perf_counter`` (a duration on one machine).
The cache lives in ``shm_dir()`` (``/dev/shm``; ``--dir`` elsewhere) under a fresh prefix that is removed at the end.

Rule (fixed before measuring, T1.5): PASS iff p95 <= 100 ms over at least 50 reads, at least 20 versions were
published during the window, and every read saw a complete version (the leaf values equal the version's). Run alone
on a quiet machine, two runs, both must pass.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import secrets
import statistics
import time
from pathlib import Path

import numpy as np

from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.node.weights import (
    WeightCacheReader,
    WeightCacheWriter,
    acquire_node_lock,
    cache_prefix,
    release_node_lock,
    shm_dir,
)

MODEL_BYTES = 50 * 2**20
LEAVES = 100
PUBLISH_SEC = 0.5
P95_MAX_SEC = 0.100
MIN_READS, MIN_VERSIONS = 50, 20
AGENT = "main"
PAGE_FLOATS = 1024                # 4 KiB of float32


def _state_dict(version: int) -> dict[str, np.ndarray]:
    per_leaf = MODEL_BYTES // LEAVES // 4
    return {f"layer{i}.weight": np.full(per_leaf, float(version), np.float32) for i in range(LEAVES)}


def _writer(root: str, prefix: str, stop, published) -> None:
    writer = WeightCacheWriter(Path(root), prefix)
    version = 0
    next_at = time.monotonic()
    while not stop.is_set():
        version += 1
        frame = encode_weights(_state_dict(version), WeightsMeta(AGENT, "latest", 1, version))
        writer.publish_latest(AGENT, frame, version=version, generation=1, teacher_active=False)
        published.value = version
        next_at += PUBLISH_SEC
        stop.wait(max(0.0, next_at - time.monotonic()))


def _touch(state_dict: dict[str, np.ndarray]) -> float:
    """Read one float of every 4 KiB page of every leaf; returns their sum."""
    return float(sum(leaf[::PAGE_FLOATS].sum(dtype=np.float64) for leaf in state_dict.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--dir", type=str, default=None, help="cache directory (default: shm_dir())")
    parser.add_argument("--json", type=str, default=None, help="write the numbers to this file")
    args = parser.parse_args()
    root = Path(args.dir) if args.dir else shm_dir()
    prefix = cache_prefix(secrets.token_hex(32), secrets.token_hex(16), 1)
    acquire_node_lock(root, prefix)
    ctx = mp.get_context("spawn")
    stop, published = ctx.Event(), ctx.Value("q", 0)
    proc = ctx.Process(target=_writer, args=(str(root), prefix, stop, published), daemon=True)
    proc.start()
    reader = WeightCacheReader(root, prefix)
    durations: list[float] = []
    incomplete = 0
    pages_per_leaf = -(-(MODEL_BYTES // LEAVES // 4) // PAGE_FLOATS)
    try:
        while reader.latest_version(AGENT) is None:
            if not proc.is_alive():
                raise SystemExit(f"the writer process died (exit code {proc.exitcode})")
            time.sleep(0.01)
        first_version = published.value
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            start = time.perf_counter()
            cached = reader.read_latest(AGENT)
            total = _touch(cached.state_dict) if cached is not None else None
            durations.append(time.perf_counter() - start)
            if cached is None or total != cached.meta.policy_version * pages_per_leaf * LEAVES:
                incomplete += 1
            del cached
        versions = published.value - first_version
    finally:
        stop.set()
        proc.join(timeout=10)
        release_node_lock(root, prefix)
    p95 = statistics.quantiles(durations, n=20)[-1] if len(durations) >= 20 else max(durations)
    ok = p95 <= P95_MAX_SEC and len(durations) >= MIN_READS and versions >= MIN_VERSIONS and incomplete == 0
    verdict = "PASS" if ok else "FAIL"
    print(f"{len(durations)} reads of {MODEL_BYTES / 2**20:.0f} MB while {versions} versions were published: "
          f"median {statistics.median(durations) * 1e3:.2f} ms, p95 {p95 * 1e3:.2f} ms, max {max(durations) * 1e3:.2f} "
          f"ms, incomplete {incomplete} (p95 <= {P95_MAX_SEC * 1e3:.0f} ms: {verdict})")
    if args.json:
        Path(args.json).write_text(json.dumps({"reads": len(durations), "versions": versions, "p95_sec": p95,
                                               "median_sec": statistics.median(durations), "max_sec": max(durations),
                                               "incomplete": incomplete, "verdict": verdict, "dir": str(root)},
                                              indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
