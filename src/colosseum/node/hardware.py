"""What a node reports about its machine (SP5 spec 2.6) and the default slots.

Cores honour the process's CPU affinity and a cgroup v2 / v1 CPU quota (containers); physical cores come from
``/proc/cpuinfo`` (pairs of ``physical id`` / ``core id``), else ``sysctl hw.physicalcpu`` (macOS), else the logical
count; both are capped by the quota. GPUs come from ``nvidia-smi`` (also ``/usr/lib/wsl/lib/nvidia-smi`` on WSL2),
without initializing CUDA in the node process; MPS from ``torch.backends.mps``. The torch version and build are
``bootstrap.torch_select.torch_identity()``'s (amendment A24: hub and node compare the same strings). Every probe is
a module-level function, so tests replace them."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Hardware", "affinity_cores", "cgroup_quota_cores", "default_slots", "machine_id", "memory_bytes",
           "nvidia_gpus", "physical_cores_raw", "probe", "torch_versions"]

_NVIDIA_SMI = ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROBE_TIMEOUT_SEC = 10


@dataclass(frozen=True)
class Hardware:
    logical_cores: int
    physical_cores: int
    memory_bytes: int
    gpus: tuple[dict, ...]          # {"index", "name", "memory_bytes", "kind": "cuda" | "mps"}
    os: str
    platform: str
    python: str
    torch_version: str
    torch_build: str


def affinity_cores() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def cgroup_quota_cores(root: Path = _CGROUP_ROOT) -> float | None:
    """The CPU quota in cores (cgroup v2 ``cpu.max``, else v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``); None
    without a quota."""
    try:
        text = (root / "cpu.max").read_text().split()
        if text and text[0] != "max":
            return int(text[0]) / int(text[1])
        if text:
            return None
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def physical_cores_raw(cpuinfo: Path = Path("/proc/cpuinfo")) -> int | None:
    """Distinct (``physical id``, ``core id``) pairs of ``cpuinfo``; ``sysctl hw.physicalcpu`` on macOS; else None."""
    try:
        pairs, phys = set(), None
        for line in cpuinfo.read_text().splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "physical id":
                phys = value.strip()
            elif key == "core id":
                pairs.add((phys, value.strip()))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True,
                                      timeout=_PROBE_TIMEOUT_SEC).stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def nvidia_gpus() -> list[dict]:
    """The GPUs ``nvidia-smi`` lists (no CUDA initialization in this process); [] without the tool."""
    for exe in _NVIDIA_SMI:
        path = shutil.which(exe) or (exe if os.path.isfile(exe) else None)
        if path is None:
            continue
        try:
            out = subprocess.run([path, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=_PROBE_TIMEOUT_SEC)
        except (OSError, subprocess.SubprocessError):
            continue
        gpus = []
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0].isdigit():
                try:
                    memory = int(float(parts[2])) << 20
                except ValueError:
                    memory = 0
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_bytes": memory, "kind": "cuda"})
        return gpus
    return []


def mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - no torch / no MPS
        return False


def torch_versions() -> tuple[str, str]:
    """``(version, build)`` of this interpreter's torch as ``bootstrap.torch_select.torch_identity`` names them
    (A24); ``("", "")`` without torch."""
    try:
        from colosseum.bootstrap.torch_select import torch_identity

        return torch_identity()
    except Exception:  # noqa: BLE001 - no (working) torch: the hub sees empty strings
        return "", ""


def probe() -> Hardware:
    logical = affinity_cores()
    quota = cgroup_quota_cores()
    if quota is not None:
        logical = max(1, min(logical, math.ceil(quota)))
    physical = physical_cores_raw() or logical
    physical = max(1, min(physical, logical))
    gpus = nvidia_gpus()
    if not gpus and sys.platform == "darwin" and mps_available():
        gpus = [{"index": 0, "name": "mps", "memory_bytes": 0, "kind": "mps"}]
    version, build = torch_versions()
    return Hardware(logical_cores=logical, physical_cores=physical, memory_bytes=memory_bytes(), gpus=tuple(gpus),
                    os=sys.platform, platform=platform.platform(), python=platform.python_version(),
                    torch_version=version, torch_build=build)


def default_slots(hw: Hardware) -> dict:
    """Spec 2.6: workers = physical cores - 1 (at least 1); one evaluator and one CPU learner on >= 4 physical
    cores; one learner slot per GPU."""
    big = hw.physical_cores >= 4
    return {"workers": max(1, hw.physical_cores - 1), "evaluators": 1 if big else 0,
            "cpu_learners": 1 if big else 0, "gpus": [int(g["index"]) for g in hw.gpus]}


def machine_id() -> str:
    """``/etc/machine-id`` (or dbus's), ``IOPlatformUUID`` on macOS, else 32 hex chars of the hostname's sha256."""
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            text = Path(path).read_text().strip()
            if text:
                return text
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], capture_output=True, text=True,
                                 timeout=_PROBE_TIMEOUT_SEC).stdout
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split("=")[-1].strip().strip('"')
        except (OSError, subprocess.SubprocessError):
            pass
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:32]
