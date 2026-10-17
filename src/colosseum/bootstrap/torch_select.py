"""The torch build of a node (SP5 spec 2.4): the hub's torch version, the build chosen by the node's hardware.

``torch_spec`` returns the requirement (``torch==<hub version>``), the uv ``--torch-backend`` value (``cu130``,
``cpu``, ``auto``; None = the default PyPI wheel, macOS) and the driver version to export as
``UV_CUDA_DRIVER_VERSION`` (with it, uv's choice does not depend on ``PATH``: in WSL2 ``nvidia-smi`` lives in
``/usr/lib/wsl/lib``, which a clean environment lacks, and uv then silently picks the CPU wheel).

Rules (``research/sp5_node_environment.md`` §1):
- macOS: the PyPI wheel (CPU / MPS); torch >= 2.3 has no Intel-Mac wheels, torch >= 2.13 needs macOS 14 on arm64;
- Linux without an NVIDIA GPU: ``cpu``;
- Linux with NVIDIA GPUs: the hub's CUDA build when the driver supports it (CUDA 13 also needs compute
  capability >= 7.5 on every GPU); else, for a GPU older than Turing (< 7.5), ``cu126`` while torch <= 2.14 and the
  driver supports CUDA 12, otherwise ``cpu`` with a warning; else ``auto`` (uv takes the newest build the driver
  supports, with a warning when that is not the hub's build);
- anything else (native Windows): ``TorchSelectError``.

Standard library only at module level and no colosseum imports: the installer embeds this file next to the
loader (T6.2). ``torch_identity`` imports torch lazily (hub and node agent only).
"""

from __future__ import annotations

import os
import platform as _platform
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

WSL_LIB = "/usr/lib/wsl/lib"
CU13_MIN_COMPUTE_CAP = (7, 5)
OLD_GPU_BACKEND = "cu126"
OLD_GPU_LAST_TORCH = (2, 14)            # PyTorch: 2.14 is the last release with CUDA 12 wheels (sm_50..sm_70)
MAC_ARM_ONLY_FROM = (2, 3)
MAC_14_FROM = (2, 13)
# Minimum Linux driver per CUDA major (NVIDIA CUDA Toolkit release notes, table 3; uv LINUX_CUDA_DRIVERS).
MIN_DRIVER = {13: (580, 0, 0), 12: (525, 60, 13), 11: (450, 80, 2)}
_QUERY = "index,name,memory.total,driver_version,compute_cap"


class TorchSelectError(ValueError):
    """The hub's torch cannot be installed on this machine (the message says why and what to do)."""


@dataclass(frozen=True)
class NvidiaGpu:
    index: int
    name: str
    memory_mib: int
    driver: str
    compute_cap: str | None = None       # "8.9"; None: the driver's nvidia-smi does not report it


@dataclass(frozen=True)
class NodePlatform:
    system: str                          # platform.system(): "Linux", "Darwin", "Windows", ...
    machine: str                         # platform.machine(): "x86_64", "arm64", "aarch64"
    mac_release: str | None = None       # platform.mac_ver()[0] on macOS


@dataclass(frozen=True)
class TorchChoice:
    requirement: str                     # "torch==2.14.1"
    backend: str | None                  # uv --torch-backend; None = the default index (macOS)
    cuda_driver: str | None              # export as UV_CUDA_DRIVER_VERSION
    warning: str | None                  # shown by the loader (and logged by the node)


def parse_version(text: str) -> tuple[int, ...]:
    """Leading numeric release components: "2.14.1+cu130" -> (2, 14, 1), "2.15.0rc1" -> (2, 15, 0)."""
    out = []
    for part in text.split("+", 1)[0].split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
        if digits != part:
            break
    return tuple(out)


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def build_tag(cuda: str | None, hip: str | None = None) -> str:
    """``torch.version.cuda`` / ``.hip`` -> ``cu130`` / ``rocm7.1`` / ``cpu``."""
    if cuda:
        return "cu" + "".join(cuda.split(".")[:2])
    if hip:
        return "rocm" + ".".join(hip.split(".")[:2])
    return "cpu"


def torch_identity() -> tuple[str, str]:
    """(public version, build tag) of the torch of this interpreter (imports torch)."""
    import torch

    return public_version(torch.__version__), build_tag(torch.version.cuda, getattr(torch.version, "hip", None))


def cuda_major(backend: str) -> int | None:
    if backend.startswith("cu") and backend[2:].isdigit() and len(backend) >= 4:
        return int(backend[2:-1])
    return None


def probe_platform() -> NodePlatform:
    mac = (_platform.mac_ver()[0] or None) if _platform.system() == "Darwin" else None
    return NodePlatform(_platform.system(), _platform.machine(), mac)


def nvidia_smi_environ(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """``environ`` with WSL2's driver directory on PATH (when it exists)."""
    env = dict(os.environ if environ is None else environ)
    if os.path.isdir(WSL_LIB) and WSL_LIB not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = os.pathsep.join(filter(None, [env.get("PATH"), WSL_LIB]))
    return env


def probe_nvidia(*, run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 environ: Mapping[str, str] | None = None) -> list[NvidiaGpu]:
    """NVIDIA GPUs from ``nvidia-smi`` (no torch); [] without a driver. An old driver that does not know
    ``compute_cap`` is asked again without it."""
    env = nvidia_smi_environ(environ)
    for query in (_QUERY, _QUERY.rsplit(",", 1)[0]):
        try:
            out = run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=20, env=env)
        except (OSError, subprocess.SubprocessError):
            return []
        if out.returncode != 0:
            continue
        gpus = []
        for line in out.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 4:
                continue
            try:
                cc = cols[4] if len(cols) > 4 and parse_version(cols[4]) else None
                gpus.append(NvidiaGpu(int(cols[0]), cols[1], int(float(cols[2])), cols[3], cc))
            except ValueError:
                continue
        return gpus
    return []


def torch_spec(hub_torch: str, hub_build: str, *, platform: NodePlatform,
               gpus: Sequence[NvidiaGpu]) -> TorchChoice:
    version = public_version(hub_torch)
    req = f"torch=={version}"
    tv = parse_version(version)
    if platform.system == "Darwin":
        if tv >= MAC_ARM_ONLY_FROM and platform.machine not in ("arm64", "aarch64"):
            raise TorchSelectError(f"torch {version} has no wheels for Intel Macs (the last one is 2.2.2); "
                                   "use an Apple Silicon Mac or another machine")
        if tv >= MAC_14_FROM and platform.mac_release and parse_version(platform.mac_release) < (14,):
            raise TorchSelectError(f"torch {version} needs macOS 14 or newer (this Mac: {platform.mac_release})")
        return TorchChoice(req, None, None, None)
    if platform.system != "Linux":
        raise TorchSelectError(f"{platform.system} is not supported for nodes (Linux, macOS, or Windows via WSL2)")
    if not gpus:
        return TorchChoice(req, "cpu", None, None)
    driver = min((g.driver for g in gpus), key=parse_version)
    dv = parse_version(driver)
    caps = [parse_version(g.compute_cap) for g in gpus if g.compute_cap]
    cc = min(caps) if caps else None
    old_gpu = cc is not None and cc < CU13_MIN_COMPUTE_CAP
    hub_major = cuda_major(hub_build)
    if hub_major is not None and dv >= MIN_DRIVER.get(hub_major, (10 ** 6,)) and not (hub_major >= 13 and old_gpu):
        return TorchChoice(req, hub_build, driver, None)
    gpu_text = f"{gpus[0].name} (driver {driver}" + (f", compute capability {'.'.join(map(str, cc))})" if cc else ")")
    if old_gpu:
        if tv[:2] <= OLD_GPU_LAST_TORCH and dv >= MIN_DRIVER[12]:
            return TorchChoice(req, OLD_GPU_BACKEND, driver,
                               f"{gpu_text} is older than Turing: using the {OLD_GPU_BACKEND} build of torch {version}"
                               + (f" instead of the hub's {hub_build}" if hub_build != OLD_GPU_BACKEND else ""))
        return TorchChoice(req, "cpu", driver,
                           f"no CUDA build of torch {version} supports {gpu_text}: this node uses the CPU build "
                           "(learners on it run on the CPU)")
    note = (f"the driver of {gpu_text} is too old for the hub's {hub_build} build of torch {version}: uv picks the "
            "newest CUDA build this driver supports (a CPU build if there is none; update the driver to avoid it)"
            if hub_major is not None else None)
    return TorchChoice(req, "auto", driver, note)
