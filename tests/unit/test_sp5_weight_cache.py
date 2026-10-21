"""The shared-memory weight cache (SP5 T1.5, spec blocks 4.6 and 7.8; §6 "Узел с поддельным хабом: кэш весов"):
segments and pointers, the newest two versions kept, readers keep their views after a deletion and re-read the
pointer when its segment vanished, snapshots and blobs by id (A18 blob keys), the free-space check, prefixes of two
hubs or runs never meet, the node lock, and cleanup of dead owners only."""
from __future__ import annotations

import errno
import json
import os
import stat
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from colosseum.core.run_state import process_start_ticks
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.net.frame import FLAG_LZ4, encode_frame
from colosseum.node import weights
from colosseum.node.weights import (
    BLOB_KINDS,
    SHM_DIR_ENV,
    CacheFullError,
    CacheKey,
    CacheLockedError,
    WeightCacheReader,
    WeightCacheWriter,
    acquire_node_lock,
    blob_key,
    cache_prefix,
    cleanup_dead,
    frame_digest,
    release_node_lock,
    shm_dir,
    split_blob_key,
)

CA = "ab" * 32
RUN = "0123456789abcdef" * 2


def _frame(agent: str, version: int, value: float, *, kind: str = "latest", generation: int = 1,
           teacher_active: bool = False) -> bytes:
    sd = {"w": np.full((4, 4), value, np.float32), "b": np.arange(3, dtype=np.int64)}
    return encode_weights(sd, WeightsMeta(agent, kind, generation, version, teacher_active=teacher_active))


def _pair(tmp_path, node: int = 1, run: str = RUN, ca: str = CA):
    prefix = cache_prefix(ca, run, node)
    return WeightCacheWriter(tmp_path / "shm", prefix), WeightCacheReader(tmp_path / "shm", prefix)


def _files(tmp_path) -> list[str]:
    return sorted(os.listdir(tmp_path / "shm"))


def test_the_cache_dir_and_the_prefix(tmp_path, monkeypatch):
    assert shm_dir() == tmp_path / "shm"                                     # tests/conftest.py redirects it
    monkeypatch.delenv(SHM_DIR_ENV)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(weights.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    assert shm_dir() == tmp_path / "tmp" / "colosseum-shm"                   # macOS: files in the temp dir
    assert cache_prefix(CA, RUN, 3) == "colosseum-abababab-01234567-n3"
    for bad in (("xyz", RUN, 1), (CA, "run-1", 1), (CA, RUN, -1), (CA + "\n", RUN, 1)):
        with pytest.raises(ValueError):
            cache_prefix(*bad)


def test_publish_and_read_the_latest_weights(tmp_path):
    writer, reader = _pair(tmp_path)
    assert reader.read_latest("main") is None and reader.latest_version("main") is None
    frame = _frame("main", 7, 1.5, generation=2, teacher_active=True)
    name, digest = writer.publish_latest("main", frame, version=7, generation=2, teacher_active=True)
    assert name == "colosseum-abababab-01234567-n1-main-v7"
    assert reader.latest_version("main") == (7, 2)
    cached = reader.read_latest("main")
    assert cached.key == CacheKey("latest", "main") and cached.meta.policy_version == 7
    assert cached.meta.generation == 2 and cached.meta.teacher_active
    assert np.all(cached.state_dict["w"] == 1.5) and cached.state_dict["b"].tolist() == [0, 1, 2]
    assert not cached.state_dict["w"].flags.writeable                         # views of a read-only mapping
    assert reader.read_latest("main", newer_than=7) is None and reader.read_latest("main", newer_than=6) is not None
    pointer = json.loads((tmp_path / "shm" / "colosseum-abababab-01234567-n1-main.latest").read_text())
    assert set(pointer) == {"segment", "version", "generation", "size", "digest", "teacher_active"}
    assert pointer["size"] == cached.nbytes and pointer["digest"] == digest == frame_digest(frame)
    assert stat.S_IMODE(os.stat(tmp_path / "shm" / name).st_mode) == 0o600


def test_the_newest_two_versions_are_kept_and_mapped_views_survive_deletion(tmp_path):
    writer, reader = _pair(tmp_path)
    for version in (1, 2, 3):
        writer.publish_latest("main", _frame("main", version, float(version)), version=version, generation=1,
                              teacher_active=False)
    held = reader.read_latest("main")                                         # maps v3
    for version in (4, 5):
        writer.publish_latest("main", _frame("main", version, float(version)), version=version, generation=1,
                              teacher_active=False)
    assert [f for f in _files(tmp_path) if "-main-v" in f] == ["colosseum-abababab-01234567-n1-main-v4",
                                                                 "colosseum-abababab-01234567-n1-main-v5"]
    assert held.meta.policy_version == 3 and np.all(held.state_dict["w"] == 3.0)   # v3 is deleted, still readable
    restarted = WeightCacheWriter(tmp_path / "shm", writer.prefix)           # a new learner incarnation adopts v4, v5
    restarted.publish_latest("main", _frame("main", 6, 6.0, generation=2), version=6, generation=2,
                             teacher_active=False)
    assert sorted(f for f in _files(tmp_path) if "-main-v" in f)[-2:] == ["colosseum-abababab-01234567-n1-main-v5",
                                                                          "colosseum-abababab-01234567-n1-main-v6"]
    assert len([f for f in _files(tmp_path) if "-main-v" in f]) == 2


def test_a_reader_re_reads_the_pointer_when_its_segment_vanished(tmp_path, monkeypatch):
    writer, reader = _pair(tmp_path)
    writer.publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1, teacher_active=False)
    real_open = weights._open_segment
    calls = []

    def racing_open(path):
        calls.append(path.name)
        if len(calls) == 1:                                                   # the writer moves on in between
            writer.publish_latest("main", _frame("main", 2, 2.0), version=2, generation=1, teacher_active=False)
            writer.publish_latest("main", _frame("main", 3, 3.0), version=3, generation=1, teacher_active=False)
            raise FileNotFoundError(path)
        return real_open(path)

    monkeypatch.setattr(weights, "_open_segment", racing_open)
    cached = reader.read_latest("main")
    assert cached.meta.policy_version == 3 and calls[-1].endswith("-main-v3") and len(calls) == 2
    (tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-v3").unlink()   # a pointer to nothing
    monkeypatch.setattr(weights, "_open_segment", real_open)
    assert reader.read_latest("main") is None                                 # bounded retries, no exception


def test_snapshots_and_blobs_by_id(tmp_path):
    writer, reader = _pair(tmp_path)
    snap, blob = CacheKey("snapshot", "main", "ckpt_v40"), blob_key("fixed", "random_net", "0f" * 16)
    assert blob == CacheKey("blob", "random_net", "fixed-" + "0f" * 16)
    assert not reader.has(snap) and reader.get(snap) is None
    assert writer.put(snap, _frame("main", 40, 4.0, kind="snapshot")) == "colosseum-abababab-01234567-n1-main-ckpt_v40"
    inode = os.stat(tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v40").st_ino
    writer.put(snap, _frame("main", 40, 9.0, kind="snapshot"))               # immutable: kept as is
    assert os.stat(tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v40").st_ino == inode
    assert writer.put(blob, _frame("random_net", 0, 0.5, kind="fixed")) == \
        "colosseum-abababab-01234567-n1-blob-fixed-" + "0f" * 16
    assert reader.has(snap) and reader.get(snap).meta.policy_version == 40
    assert np.all(reader.get(snap).state_dict["w"] == 4.0) and reader.get(blob).meta.kind == "fixed"
    writer.drop(snap)
    assert not reader.has(snap) and reader.has(blob)
    bad_keys = (CacheKey("snapshot", "main", "v40"), CacheKey("blob", "", "NOT-HEX"),
                CacheKey("blob", "", "0f" * 16),                               # A18: a blob ident names its kind
                CacheKey("blob", "", "recovery-0f"), CacheKey("blob", "", "fixed-../x"),
                CacheKey("blob", "", "fixed-" + "a" * 161), CacheKey("snapshot", "main", "ckpt_v1\n"),
                CacheKey("snapshot", "a b", "ckpt_v1"), CacheKey("snapshot", "main\n", "ckpt_v1"))
    for bad in bad_keys:
        with pytest.raises(ValueError):
            writer.put(bad, b"")
    with pytest.raises(ValueError):
        writer.put(CacheKey("latest", "main"), b"")


def test_blob_keys_round_trip_through_split_blob_key():
    assert BLOB_KINDS == ("fixed", "teacher", "init", "eval")
    for kind in BLOB_KINDS:
        for blob_id in ("0f" * 16, "bc_net.v2-final", "a" * 160):
            key = blob_key(kind, "main", blob_id)
            assert key.kind == "blob" and key.agent_id == "main" and split_blob_key(key) == (kind, blob_id)
    for bad in (("recovery", "main", "0f"), ("snapshot", "main", "0f"), ("fixed", "main", ""),
                ("fixed", "main", ".hidden"), ("fixed", "main", "a/b"), ("fixed", "main", "a" * 161)):
        with pytest.raises(ValueError):
            blob_key(*bad)
    for bad_key in (CacheKey("blob", "", "0f" * 16), CacheKey("snapshot", "main", "ckpt_v1"),
                    CacheKey("blob", "", "eval-")):
        with pytest.raises(ValueError):
            split_blob_key(bad_key)


def test_a_corrupt_or_mismatched_segment_is_ignored_with_one_warning(tmp_path, caplog):
    writer, reader = _pair(tmp_path)
    (tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v1").write_bytes(b"garbage" * 10)
    assert reader.get(CacheKey("snapshot", "main", "ckpt_v1")) is None
    assert reader.get(CacheKey("snapshot", "main", "ckpt_v1")) is None
    writer.publish_latest("main", _frame("other", 5, 1.0), version=5, generation=1, teacher_active=False)
    assert reader.read_latest("main") is None
    assert len([r for r in caplog.records if "ignored" in r.getMessage()]) == 2


def test_an_unreadable_segment_reads_as_absent_with_one_warning_per_errno(tmp_path, monkeypatch, caplog):
    """P-PF1-9: a missing (or empty) segment is a silent None; any other I/O error is a None with one WARNING per
    (segment, errno), so a held lineup always has a log line."""
    writer, reader = _pair(tmp_path)
    snap = CacheKey("snapshot", "main", "ckpt_v2")
    writer.put(snap, _frame("main", 2, 2.0, kind="snapshot"))
    writer.publish_latest("main", _frame("main", 3, 3.0), version=3, generation=1, teacher_active=False)
    (tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v9").write_bytes(b"")   # empty: silent
    assert reader.get(CacheKey("snapshot", "main", "ckpt_v9")) is None
    assert reader.get(CacheKey("snapshot", "main", "ckpt_v8")) is None                    # missing: silent
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    failure = {"errno": errno.EACCES}

    def failing_open(path):
        raise OSError(failure["errno"], os.strerror(failure["errno"]), str(path))

    monkeypatch.setattr(weights, "_open_segment", failing_open)
    for _ in range(3):
        assert reader.get(snap) is None and reader.read_latest("main") is None
    failure["errno"] = errno.EMFILE
    assert reader.get(snap) is None
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warned) == 3 and all("cannot read" in m for m in warned)
    assert sum("ckpt_v2" in m for m in warned) == 2 and sum("main-v3" in m for m in warned) == 1


def test_the_free_space_check_refuses_a_segment_with_the_hint(tmp_path, monkeypatch):
    writer, _ = _pair(tmp_path)
    frame = _frame("main", 1, 1.0)
    usage = weights.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(weights.shutil, "disk_usage",
                        lambda path: usage._replace(free=2 * len(frame) + weights.SPACE_MARGIN_BYTES - 1))
    with pytest.raises(CacheFullError, match="--shm-size"):
        writer.publish_latest("main", frame, version=1, generation=1, teacher_active=False)
    with pytest.raises(CacheFullError, match="emptyDir"):
        writer.put(blob_key("fixed", "main", "aa"), frame)
    assert not any(f.endswith("-main-v1") or "-blob-" in f for f in _files(tmp_path))
    monkeypatch.setattr(weights.shutil, "disk_usage",
                        lambda path: usage._replace(free=2 * len(frame) + weights.SPACE_MARGIN_BYTES))
    writer.publish_latest("main", frame, version=1, generation=1, teacher_active=False)


def test_two_threads_writing_one_name_never_share_a_temp_file(tmp_path):
    """P-PF1-6: the node writes segments from a transfer pool; the temp name is per thread, not only per process."""
    root = tmp_path / "shm"
    target = root / "colosseum-abababab-01234567-n1-main-ckpt_v1"
    payloads = [bytes([i]) * 200_000 for i in range(8)]
    barrier, errors = threading.Barrier(len(payloads)), []

    def write(data: bytes) -> None:
        barrier.wait()
        try:
            for _ in range(20):
                weights._write_file(target, data)
        except Exception as e:                                                 # pragma: no cover - the failure
            errors.append(e)

    threads = [threading.Thread(target=write, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and target.read_bytes() in payloads
    assert os.listdir(root) == [target.name]                                   # no temp file left behind


def test_two_hubs_and_two_runs_on_one_machine_never_share_files(tmp_path):
    a, ra = _pair(tmp_path)
    b, rb = _pair(tmp_path, ca="cd" * 32)
    c, rc = _pair(tmp_path, run="fedcba9876543210" * 2)
    d, rd = _pair(tmp_path, node=12)
    for i, writer in enumerate((a, b, c, d)):
        writer.publish_latest("main", _frame("main", i + 1, float(i)), version=i + 1, generation=1,
                              teacher_active=False)
    assert [r.latest_version("main")[0] for r in (ra, rb, rc, rd)] == [1, 2, 3, 4]
    assert len(_files(tmp_path)) == 8 and len({f.split("-main")[0] for f in _files(tmp_path)}) == 4


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_the_node_lock_and_cleanup_of_dead_owners_only(tmp_path):
    root = tmp_path / "shm"
    mine, theirs = cache_prefix(CA, RUN, 1), cache_prefix(CA, RUN, 2)
    lock = acquire_node_lock(root, mine)
    assert json.loads(lock.read_text()) == {"pid": os.getpid(), "proc_start": process_start_ticks(os.getpid())}
    assert acquire_node_lock(root, mine) == lock                               # re-acquiring our own lock is fine
    WeightCacheWriter(root, mine).publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1,
                                                 teacher_active=False)
    dead = WeightCacheWriter(root, theirs)
    (root / f"{theirs}.lock").write_text(json.dumps({"pid": _dead_pid(), "proc_start": None}))
    dead.publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1, teacher_active=False)
    (root / f".{theirs}-main-v2.tmp-999").write_bytes(b"half")                  # a dead writer's temp file
    stray = cache_prefix(CA, RUN, 3)
    (root / f"{stray}-main-v9").write_bytes(b"x")                              # no lock at all
    (root / "unrelated-file").write_bytes(b"keep")
    reused = cache_prefix(CA, RUN, 4)                                          # the pid lives, the start time differs
    (root / f"{reused}.lock").write_text(json.dumps({"pid": os.getpid(), "proc_start": -1}))
    removed = cleanup_dead(root)
    assert sorted(os.listdir(root)) == sorted([f"{mine}.lock", f"{mine}-main-v1", f"{mine}-main.latest",
                                               "unrelated-file"])
    assert f"{theirs}.lock" in removed and f"{stray}-main-v9" in removed and f"{reused}.lock" in removed
    assert cleanup_dead(root) == [] and cleanup_dead(tmp_path / "missing") == []
    parent = os.getppid()                                                     # a live foreign owner
    (root / f"{theirs}.lock").write_text(json.dumps({"pid": parent, "proc_start": process_start_ticks(parent)}))
    with pytest.raises(CacheLockedError, match=str(parent)):
        acquire_node_lock(root, theirs)
    (root / f"{reused}.lock").write_text(json.dumps({"pid": _dead_pid(), "proc_start": None}))
    assert json.loads(acquire_node_lock(root, reused).read_text())["pid"] == os.getpid()   # a dead owner's lock


def test_release_removes_only_this_prefix_and_never_a_live_foreign_owners_files(tmp_path):
    root = tmp_path / "shm"
    mine, other = cache_prefix(CA, RUN, 1), cache_prefix(CA, RUN, 11)
    acquire_node_lock(root, mine)
    writer = WeightCacheWriter(root, mine)
    writer.publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1, teacher_active=False)
    writer.put(blob_key("teacher", "main", "t1"), _frame("main", 0, 0.0, kind="teacher"))
    WeightCacheWriter(root, other).publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1,
                                                  teacher_active=False)
    release_node_lock(root, mine)
    assert all(f.startswith(other) for f in os.listdir(root)) and len(os.listdir(root)) == 2
    release_node_lock(tmp_path / "missing", mine)                              # nothing there: no error
    parent = os.getppid()
    (root / f"{other}.lock").write_text(json.dumps({"pid": parent, "proc_start": process_start_ticks(parent)}))
    release_node_lock(root, other)                                             # not ours: kept, WARNING only
    assert len(os.listdir(root)) == 3


# ---------------------------------------------------------------------------
# Fix round 1 (T1.5 review)
# ---------------------------------------------------------------------------


def test_agent_ids_blob_and_blob_dash_are_reserved(tmp_path):
    writer, reader = _pair(tmp_path)
    for agent in ("blob", "blob-fixed", "blob-x"):
        with pytest.raises(ValueError, match="reserved"):
            writer.publish_latest(agent, _frame(agent, 1, 1.0), version=1, generation=1, teacher_active=False)
        with pytest.raises(ValueError, match="reserved"):
            writer.put(CacheKey("snapshot", agent, "ckpt_v1"), _frame(agent, 1, 1.0, kind="snapshot"))
        with pytest.raises(ValueError, match="reserved"):
            reader.read_latest(agent)
    writer.publish_latest("blobby", _frame("blobby", 1, 1.0), version=1, generation=1, teacher_active=False)
    writer.publish_latest("blob_net", _frame("blob_net", 1, 1.0), version=1, generation=1, teacher_active=False)
    assert reader.read_latest("blobby").meta.agent_id == "blobby"
    assert not any("-blob-" in f for f in _files(tmp_path))


def test_a_writer_handover_leaks_no_segments(tmp_path):
    """The node publishes fetched versions while the learner is remote, the learner publishes once it runs here (and
    the other way round): every publish prunes on disk, so at most KEEP_LATEST segments of the agent remain."""
    node_writer, reader = _pair(tmp_path)
    learner_writer = WeightCacheWriter(tmp_path / "shm", node_writer.prefix)
    for version in range(1, 9):
        writer = (node_writer, learner_writer)[(version // 2) % 2]          # two publishes each, alternating
        writer.publish_latest("main", _frame("main", version, float(version)), version=version, generation=1,
                              teacher_active=False)
        segments = [f for f in _files(tmp_path) if "-main-v" in f]
        assert len(segments) <= weights.KEEP_LATEST and f"{node_writer.prefix}-main-v{version}" in segments
        assert reader.read_latest("main").meta.policy_version == version
    assert sorted(f for f in _files(tmp_path) if "-main-v" in f) == [f"{node_writer.prefix}-main-v7",
                                                                     f"{node_writer.prefix}-main-v8"]


_ACQUIRER = """
import sys, time
from pathlib import Path
from colosseum.node import weights
root, prefix, work, index = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
real_write = weights._write_file
def slow_write(path, data):
    if path.name.endswith('.lock'):
        time.sleep(0.3)                     # widen the take-over window: without exclusivity both would win
    real_write(path, data)
weights._write_file = slow_write
(work / f'ready-{index}').write_text('1')
deadline = time.monotonic() + 30
while not (work / 'go').exists() and time.monotonic() < deadline:
    time.sleep(0.001)
try:
    weights.acquire_node_lock(root, prefix)
    result = 'won'
except weights.CacheLockedError:
    result = 'locked'
(work / f'result-{index}').write_text(result)
while not (work / 'stop').exists() and time.monotonic() < deadline:
    time.sleep(0.01)
"""


def test_two_processes_taking_over_a_dead_lock_exactly_one_wins(tmp_path):
    root, work = tmp_path / "shm", tmp_path / "work"
    work.mkdir()
    prefix = cache_prefix(CA, RUN, 5)
    (root / f"{prefix}.lock").write_text(json.dumps({"pid": _dead_pid(), "proc_start": None}))
    procs = [subprocess.Popen([sys.executable, "-c", _ACQUIRER, str(root), prefix, str(work), str(i)])
             for i in range(2)]
    try:
        deadline = time.monotonic() + 60
        while not all((work / f"ready-{i}").exists() for i in range(2)):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.01)
        (work / "go").write_text("1")
        while not all((work / f"result-{i}").exists() for i in range(2)):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.01)
        results = sorted((work / f"result-{i}").read_text() for i in range(2))
        assert results == ["locked", "won"]
        owner = json.loads((root / f"{prefix}.lock").read_text())["pid"]
        assert owner in [p.pid for p in procs]
    finally:
        (work / "stop").write_text("1")
        for p in procs:
            p.wait(timeout=30)


def test_a_corrupt_or_empty_pointer_is_ignored_with_one_warning_per_file(tmp_path, caplog):
    _, reader = _pair(tmp_path)
    shm = tmp_path / "shm"
    (shm / "colosseum-abababab-01234567-n1-main.latest").write_bytes(b"")
    (shm / "colosseum-abababab-01234567-n1-other.latest").write_bytes(b"{not json")
    (shm / "colosseum-abababab-01234567-n1-third.latest").write_text(json.dumps({"segment": "/etc/passwd",
                                                                                 "version": 1, "generation": 1}))
    for _ in range(3):
        assert reader.read_latest("main") is None and reader.latest_version("other") is None
        assert reader.read_latest("third") is None
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warned) == 3 and all("ignored" in m for m in warned)


def test_a_failed_pointer_write_removes_the_new_segment(tmp_path, monkeypatch):
    writer, reader = _pair(tmp_path)
    writer.publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1, teacher_active=False)
    real_write = weights._write_file

    def failing_pointer(path, data):
        if path.name.endswith(".latest"):
            raise OSError(errno.ENOSPC, "no space left", str(path))
        real_write(path, data)

    monkeypatch.setattr(weights, "_write_file", failing_pointer)
    with pytest.raises(OSError, match="no space"):
        writer.publish_latest("main", _frame("main", 2, 2.0), version=2, generation=1, teacher_active=False)
    assert [f for f in _files(tmp_path) if "-main-v" in f] == ["colosseum-abababab-01234567-n1-main-v1"]
    assert reader.read_latest("main").meta.policy_version == 1


def test_a_stale_temp_file_or_symlink_is_replaced_not_followed(tmp_path):
    root = tmp_path / "shm"
    target = root / "colosseum-abababab-01234567-n1-main-ckpt_v1"
    victim = tmp_path / "victim"
    victim.write_bytes(b"precious")
    os.symlink(victim, weights._temp_path(target))                            # planted at our temp name
    weights._write_file(target, b"payload")
    assert target.read_bytes() == b"payload" and victim.read_bytes() == b"precious"
    weights._temp_path(target).write_bytes(b"half of a crashed write")
    weights._write_file(target, b"again")
    assert target.read_bytes() == b"again" and os.listdir(root) == [target.name]


def test_the_writer_refuses_lz4_frames(tmp_path):
    writer, _ = _pair(tmp_path)
    sd = {"w": np.zeros(4096, np.float32)}
    meta = {"agent_id": "main", "kind": "latest", "generation": 1, "policy_version": 1, "teacher_active": False,
            "checkpoint_id": None, "architecture": None, "roles": None}
    compressed = encode_frame(sd, kind="weights", meta=meta, compress=True)
    assert int.from_bytes(compressed[6:8], "little") & FLAG_LZ4
    with pytest.raises(ValueError, match="plain frames"):
        writer.publish_latest("main", compressed, version=1, generation=1, teacher_active=False)
    with pytest.raises(ValueError, match="plain frames"):
        writer.put(blob_key("fixed", "main", "z1"), compressed)
    assert _files(tmp_path) == []


def test_a_held_directory_lock_is_waited_for_with_a_bound_and_named(tmp_path, monkeypatch):
    import fcntl

    root = tmp_path / "shm"
    root.mkdir(exist_ok=True)
    monkeypatch.setattr(weights, "DIR_LOCK_TIMEOUT_SEC", 0.3)
    fd = os.open(root, os.O_RDONLY)                    # another open file description: flock conflicts with it
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        start = time.monotonic()
        with pytest.raises(CacheLockedError, match=str(root)):
            acquire_node_lock(root, cache_prefix(CA, RUN, 1))
        assert time.monotonic() - start < 5.0
        fcntl.flock(fd, fcntl.LOCK_UN)
        assert acquire_node_lock(root, cache_prefix(CA, RUN, 1)).exists()   # free again: taken at once
    finally:
        os.close(fd)
