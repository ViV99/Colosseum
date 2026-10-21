"""Node hardware (SP5 T2.4; spec 2.6, 7.7): cgroup v2 / v1 quotas, physical cores, the probe honouring quota and
affinity, default slots, a stable machine id, bounded log tails."""
from __future__ import annotations

import math

from colosseum.node import hardware
from colosseum.node.hardware import Hardware, cgroup_quota_cores, default_slots, physical_cores_raw
from colosseum.node.logs import LOG_TAIL_BYTES, log_tail


def _hw(physical, gpus=()):
    return Hardware(physical * 2, physical, 1 << 30, tuple(gpus), "linux", "p", "3.12", "2.6.0", "cpu")


def test_cgroup_quotas_v2_and_v1(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert math.isclose(cgroup_quota_cores(tmp_path), 2.5)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_quota_cores(tmp_path) is None
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("150000")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert math.isclose(cgroup_quota_cores(tmp_path), 1.5)
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1")
    assert cgroup_quota_cores(tmp_path) is None


def test_physical_cores_count_core_ids_per_socket(tmp_path):
    info = tmp_path / "cpuinfo"
    blocks = [f"processor: {i}\nphysical id: {i // 8}\ncore id: {(i % 8) // 2}\n" for i in range(16)]
    info.write_text("\n".join(blocks))
    assert physical_cores_raw(info) == 8


def test_probe_honours_the_quota_and_affinity(monkeypatch):
    monkeypatch.setattr(hardware, "affinity_cores", lambda: 16)
    monkeypatch.setattr(hardware, "cgroup_quota_cores", lambda: 2.5)
    monkeypatch.setattr(hardware, "physical_cores_raw", lambda: 8)
    monkeypatch.setattr(hardware, "nvidia_gpus", lambda: [{"index": 0, "name": "x", "memory_bytes": 1, "kind": "cuda"}])
    hw = hardware.probe()
    assert (hw.logical_cores, hw.physical_cores) == (3, 3) and hw.gpus[0]["kind"] == "cuda"


def test_default_slots_follow_the_spec():
    assert default_slots(_hw(1)) == {"workers": 1, "evaluators": 0, "cpu_learners": 0, "gpus": []}
    assert default_slots(_hw(3)) == {"workers": 2, "evaluators": 0, "cpu_learners": 0, "gpus": []}
    assert default_slots(_hw(8, [{"index": 0}, {"index": 1}])) == {"workers": 7, "evaluators": 1, "cpu_learners": 1,
                                                                     "gpus": [0, 1]}


def test_machine_id_is_stable_and_the_log_tail_is_bounded(tmp_path):
    assert hardware.machine_id() == hardware.machine_id() and hardware.machine_id()
    log = tmp_path / "worker-0.log"
    assert log_tail(log) == b""
    log.write_bytes(b"a" * 10 + b"b" * LOG_TAIL_BYTES)
    assert log_tail(log) == b"b" * LOG_TAIL_BYTES and log_tail(log, 4) == b"bbbb"


def test_the_torch_identity_is_the_bootstrap_rule():
    from colosseum.bootstrap.torch_select import torch_identity

    assert hardware.torch_versions() == torch_identity()                       # A24
