"""The evaluator (SP5 T5.1): its loop in a thread over a Pipe (cheap) and one real process (connection to the
node listener, the shm cache, kill -9 without blocking the node side)."""
from __future__ import annotations

import multiprocessing as mp
import os
import select
import signal
import threading
import time
import uuid

import pytest

from colosseum.core.registry import env_spec, make_env
from colosseum.hub.evaluation import check_block_results
from colosseum.net.local import (
    ChildHello,
    EvalCancel,
    EvalDone,
    EvalFailed,
    EvalTask,
    Stop,
    address,
    listener,
    listener_socket,
)
from colosseum.node.evaluator_main import CachePlayerPool, evaluator_loop, evaluator_target
from colosseum.utils.logging import ENV_PROCESS_NAME
from eval_kit import EvalCache, FakeBlobs, fixed_players, gauntlet_tasks, snapshot_player
from game_helpers import make_test_config, scripted_agent

AGENTS = {"main": {}, "rnd": scripted_agent(), "ones": scripted_agent("game_helpers.ConstantBot", value=1)}


def _cfg(agents=AGENTS):
    return make_test_config("turns", agents=agents)


def _filled(tasks, blobs, node=1):
    cache = EvalCache(node)
    for task in tasks:
        for p in task.players:
            if p.blob is not None and not cache.has(p.blob):
                cache.put_from(blobs, p.blob)
    return cache


def _loop(cfg, cache):
    parent, child = mp.Pipe()
    thread = threading.Thread(target=evaluator_loop, args=(child,), daemon=True, kwargs=dict(
        spec=env_spec(cfg), reader=cache.reader, env_fn=lambda: make_env(cfg),
        max_idle_steps=cfg.env.max_idle_steps, node=3, index=0))
    thread.start()
    return parent, thread


def _tasks(cfg, tmp_path, *, version=3, blocks=2, owner="tournament:t", field=("rnd", "ones"), blobs=None):
    cand = snapshot_player(cfg, tmp_path / f"snap{version}", "main", version, seed=version)
    return gauntlet_tasks(cfg, cand, fixed_players(cfg, field), blocks=blocks, owner=owner, blobs=blobs)


def test_the_evaluator_plays_whole_blocks_from_the_cache(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blobs=blobs)
    conn, thread = _loop(cfg, _filled(tasks, blobs))
    conn.send(EvalTask("L1", "tournament", tasks, 2, False))
    assert conn.poll(60)
    done = conn.recv()
    assert isinstance(done, EvalDone) and not done.cancelled and set(done.results) == {t.block_id for t in tasks}
    for task in tasks:
        assert check_block_results(task, done.results[task.block_id], spec=env_spec(cfg)) is None
        assert [r.uid for r in done.results[task.block_id]] == sorted(lu.uid for lu in task.lineups)
    conn.send(Stop("end"))
    thread.join(10)
    assert not thread.is_alive()


def test_missing_weights_fail_the_lease_and_the_loop_goes_on(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blobs=blobs)
    other = _tasks(cfg, tmp_path, version=4, owner="tournament:u", blobs=blobs)    # not in the cache
    conn, thread = _loop(cfg, _filled(tasks, blobs))
    conn.send(EvalTask("L2", "tournament", other, 2, False))
    assert conn.poll(30)
    failed = conn.recv()
    assert isinstance(failed, EvalFailed) and failed.kind == "missing_weights" and "u-main-ckpt_v4" in failed.message
    conn.send(EvalTask("L3", "tournament", tasks, 2, False))
    assert conn.poll(60) and isinstance(conn.recv(), EvalDone)
    conn.send(Stop("end"))
    thread.join(10)


def test_a_cancel_stops_the_lease_mid_play(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blocks=40, blobs=blobs)
    conn, thread = _loop(cfg, _filled(tasks, blobs))
    conn.send(EvalTask("L1", "tournament", tasks, 2, False))
    conn.send(EvalCancel("L1"))
    assert conn.poll(30)
    answer = conn.recv()
    assert isinstance(answer, EvalDone) and answer.cancelled and answer.lease_id == "L1" and answer.results == {}
    conn.send(Stop("end"))
    thread.join(10)


def test_a_player_error_is_typed_and_the_loop_survives(tmp_path, monkeypatch):
    agents = {"main": {}, "bad": scripted_agent("game_helpers.PulseFailBot")}
    cfg, blobs = _cfg(agents), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blocks=1, field=("bad",), blobs=blobs)
    monkeypatch.setenv(ENV_PROCESS_NAME, "evaluator-0")         # PulseFailBot fails in evaluator processes
    conn, thread = _loop(cfg, _filled(tasks, blobs))
    conn.send(EvalTask("L1", "tournament", tasks, 1, False))
    assert conn.poll(60)
    failed = conn.recv()
    assert isinstance(failed, EvalFailed) and failed.kind == "player"
    assert "pulse failure on purpose" in failed.message and "node 3, evaluator 0" in failed.message
    assert failed.where.startswith(("match ", "block "))
    conn.send(EvalTask("L2", "tournament", tasks, 1, False))      # the loop keeps serving leases
    assert conn.poll(60) and isinstance(conn.recv(), EvalFailed)
    conn.send(Stop("end"))
    thread.join(10)
    assert not thread.is_alive()


def test_a_closed_connection_ends_the_loop():
    cfg = _cfg()
    conn, thread = _loop(cfg, EvalCache())
    conn.close()
    thread.join(10)
    assert not thread.is_alive()


def test_the_pool_keys_models_by_blob(tmp_path):
    cfg, blobs = _cfg(), FakeBlobs()
    first = _tasks(cfg, tmp_path / "a", owner="tournament:a", blobs=blobs)
    second = _tasks(cfg, tmp_path / "b", owner="tournament:b", blobs=blobs)    # the same seat names, other weights
    cache = _filled(first + second, blobs)
    pool = CachePlayerPool(env_spec(cfg), cache.reader)
    pool.reserve(first[0].players)
    one = pool.get("main", "ckpt_v3")
    pool.reserve(second[0].players)
    two = pool.get("main", "ckpt_v3")
    assert one is not two and pool.loads == 2 and pool.get("nobody", "fixed") is None
    pool.reserve(first[0].players)
    assert pool.get("main", "ckpt_v3") is one and pool.loads == 2               # kept (LRU)


def test_the_evaluator_process_connects_plays_and_dies_without_blocking_the_node(tmp_path):
    """One real process: ChildHello on the node listener, a lease played from the shm cache, then kill -9 in the
    middle of a long lease: the node side sees EOF at once (no shared lock, spec 4.7 / 7.1)."""
    cfg, blobs = _cfg(), FakeBlobs()
    tasks = _tasks(cfg, tmp_path, blobs=blobs)
    long = _tasks(cfg, tmp_path, version=5, blocks=400, owner="tournament:long", blobs=blobs)
    cache = _filled(tasks + long, blobs)
    authkey = os.urandom(32)
    name = address("ab" * 4, uuid.uuid4().hex[:8], 1, "node")          # random: abstract names are machine-global
    lst = listener(name, authkey)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=evaluator_target, daemon=True, kwargs=dict(
        name="evaluator-0", index=0, incarnation=0, node=1, node_address=name, authkey=authkey,
        cache_root=str(cache.root), cache_prefix=cache.prefix, config=cfg, log_dir=str(tmp_path / "logs")))
    proc.start()
    try:
        ready, _w, _x = select.select([listener_socket(lst)], [], [], 60)
        assert ready, f"the evaluator never connected (exit code {proc.exitcode}; see {tmp_path / 'logs'})"
        conn = lst.accept()
        hello = conn.recv()
        assert isinstance(hello, ChildHello) and (hello.role, hello.channel, hello.name) == \
            ("evaluator", "control", "evaluator-0")
        conn.send(EvalTask("L1", "tournament", tasks, 2, False))
        assert conn.poll(120) and isinstance(conn.recv(), EvalDone)
        conn.send(EvalTask("L2", "tournament", long, 2, False))
        time.sleep(1.0)
        os.kill(proc.pid, signal.SIGKILL)
        start = time.monotonic()
        with pytest.raises((EOFError, ConnectionResetError, OSError)):
            while time.monotonic() - start < 10:
                if conn.poll(0.5):
                    conn.recv()
        assert time.monotonic() - start < 10
        proc.join(10)
        assert proc.exitcode == -signal.SIGKILL
        assert "evaluator-0" in (tmp_path / "logs" / "evaluator-0.log").read_text()
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        lst.close()
