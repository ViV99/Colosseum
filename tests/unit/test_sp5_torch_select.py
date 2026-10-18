"""Torch build per node (SP5 T6.1, spec 2.4): the hub's version, the build by the node's hardware (fake probes)."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from colosseum.bootstrap import torch_select as T
from colosseum.bootstrap.torch_select import NodePlatform, NvidiaGpu, TorchSelectError, torch_spec

LINUX = NodePlatform("Linux", "x86_64")


def gpu(driver="610.88", cc="8.9", name="RTX 4070 Ti"):
    return NvidiaGpu(0, name, 12282, driver, cc)


@pytest.mark.parametrize("hub_build, gpus, backend, driver, warns", [
    ("cu130", [], "cpu", None, False),                                         # Linux without a GPU
    ("cpu", [], "cpu", None, False),
    ("cu130", [gpu()], "cu130", "610.88", False),                               # the hub's build
    ("cu126", [gpu("535.54.03")], "cu126", "535.54.03", False),
    ("cu130", [gpu("535.54.03")], "auto", "535.54.03", True),                   # driver too old for CUDA 13
    ("cpu", [gpu()], "auto", "610.88", False),                                  # a CPU hub, a GPU node
    ("cu130", [gpu("610.88", "6.1", "GTX 1080 Ti")], "cu126", "610.88", True),  # Pascal: CUDA 12.6 build
    ("cu126", [gpu("610.88", "6.1")], "cu126", "610.88", False),
    ("cu130", [gpu("470.1", "6.1")], "cpu", "470.1", True),                     # Pascal + driver below CUDA 12
    ("cu130", [gpu("610.88", None)], "cu130", "610.88", False),                 # no compute_cap reported
    ("cu130", [gpu(cc="8.9"), gpu(cc="7.0")], "cu126", "610.88", True),         # the oldest GPU decides
])
def test_linux_choice(hub_build, gpus, backend, driver, warns):
    choice = torch_spec("2.14.1+cu130", hub_build, platform=LINUX, gpus=gpus)
    assert choice.requirement == "torch==2.14.1"                              # never the local +cuXXX tag
    assert (choice.backend, choice.cuda_driver) == (backend, driver)
    assert (choice.warning is not None) == warns


def test_old_gpu_after_the_last_cuda12_torch_gets_cpu():
    choice = torch_spec("2.15.0", "cu130", platform=LINUX, gpus=[gpu("610.88", "7.0")])
    assert choice.backend == "cpu" and "CPU build" in choice.warning


def test_macos():
    mac = NodePlatform("Darwin", "arm64", "14.7.2")
    assert torch_spec("2.14.1", "cu130", platform=mac, gpus=[]) == T.TorchChoice("torch==2.14.1", None, None, None)
    with pytest.raises(TorchSelectError, match="Intel Macs"):
        torch_spec("2.14.1", "cpu", platform=NodePlatform("Darwin", "x86_64", "13.0"), gpus=[])
    with pytest.raises(TorchSelectError, match="macOS 14"):
        torch_spec("2.14.1", "cpu", platform=NodePlatform("Darwin", "arm64", "13.6"), gpus=[])
    assert torch_spec("2.12.0", "cpu", platform=NodePlatform("Darwin", "arm64", "13.6"), gpus=[]).backend is None
    with pytest.raises(TorchSelectError, match="WSL2"):
        torch_spec("2.14.1", "cpu", platform=NodePlatform("Windows", "AMD64"), gpus=[])


def test_build_tag_and_versions():
    assert T.build_tag("13.0") == "cu130" and T.build_tag("12.6.3") == "cu126"
    assert T.build_tag(None, "7.1.25") == "rocm7.1" and T.build_tag(None) == "cpu"
    assert T.parse_version("2.14.1+cu130") == (2, 14, 1) and T.parse_version("525.60.13") == (525, 60, 13)
    assert T.parse_version("2.15.0rc1") == (2, 15, 0) and T.cuda_major("cu130") == 13 and T.cuda_major("cpu") is None


def _smi(outputs):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw["env"]["PATH"]))
        rc, out = outputs[len(calls) - 1]
        return subprocess.CompletedProcess(cmd, rc, out, "")
    return run, calls


def test_probe_nvidia_parses_and_retries_without_compute_cap(monkeypatch):
    run, calls = _smi([(0, "0, NVIDIA GeForce RTX 4070 Ti, 12282, 610.88, 8.9\n"
                           "1, Tesla V100, 16384, 535.54.03, 7.0\n")])
    assert T.probe_nvidia(run=run, environ={"PATH": "/usr/bin"}) == [
        NvidiaGpu(0, "NVIDIA GeForce RTX 4070 Ti", 12282, "610.88", "8.9"),
        NvidiaGpu(1, "Tesla V100", 16384, "535.54.03", "7.0")]
    run, calls = _smi([(6, "Field compute_cap is not a valid field"), (0, "0, Old, 4096, 470.1\n")])
    assert T.probe_nvidia(run=run, environ={"PATH": "/usr/bin"}) == [NvidiaGpu(0, "Old", 4096, "470.1", None)]
    assert "compute_cap" in calls[0][0][1] and "compute_cap" not in calls[1][0][1]

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")
    assert T.probe_nvidia(run=missing, environ={}) == []


def test_wsl_driver_dir_goes_on_path(monkeypatch):
    monkeypatch.setattr(T.os.path, "isdir", lambda p: p == T.WSL_LIB)
    assert T.nvidia_smi_environ({"PATH": "/usr/bin"})["PATH"] == f"/usr/bin:{T.WSL_LIB}"
    assert T.nvidia_smi_environ({"PATH": f"/usr/bin:{T.WSL_LIB}"})["PATH"] == f"/usr/bin:{T.WSL_LIB}"


def test_module_is_standalone():
    tree = ast.parse(Path(T.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            assert all(n.split(".")[0] == "__future__" or n.split(".")[0] in sys.stdlib_module_names for n in names)
    out = subprocess.run([sys.executable, "-I", "-c", "import importlib.util, sys; s = importlib.util."
                          "spec_from_file_location('t', sys.argv[1]); m = importlib.util.module_from_spec(s); "
                          "sys.modules['t'] = m; s.loader.exec_module(m); print('torch' in sys.modules)", T.__file__],
                         capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


def test_warning_names_the_oldest_gpu():
    choice = torch_spec("2.14.1", "cu130", platform=LINUX,
                        gpus=[gpu(cc="8.9", name="RTX 4070 Ti"), gpu(cc="6.1", name="GTX 1080 Ti")])
    assert choice.backend == "cu126" and "GTX 1080 Ti" in choice.warning and "RTX 4070 Ti" not in choice.warning
    assert "compute capability 6.1" in choice.warning


def test_probe_nvidia_keeps_a_gpu_without_memory_total():
    run, _ = _smi([(0, "0, NVIDIA A100-SXM4-40GB MIG, [N/A], 580.65.06, 8.0\n")])
    assert T.probe_nvidia(run=run, environ={"PATH": "/usr/bin"}) == [
        NvidiaGpu(0, "NVIDIA A100-SXM4-40GB MIG", 0, "580.65.06", "8.0")]
