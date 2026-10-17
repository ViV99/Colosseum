"""Node requirements (SP5 T6.1, spec 2.4): the hub environment's exact versions minus torch, the CUDA / NVIDIA
packages, colosseum and local-path installs; the distributed.requirements override; the hash."""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
from pathlib import Path

import pytest

from colosseum.bootstrap.requirements import (
    EXCLUDED,
    canonical_name,
    hub_freeze,
    node_requirements,
    read_requirements_file,
    requirements_hash,
)


def _site(tmp_path: Path, dists: dict[str, tuple[str, dict | None]]) -> list[md.Distribution]:
    site = tmp_path / "site"
    for name, (version, direct_url) in dists.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        if direct_url is not None:
            (info / "direct_url.json").write_text(json.dumps(direct_url))
    return list(md.distributions(path=[str(site)]))


def test_freeze_excludes_hardware_packages_colosseum_and_local_paths(tmp_path):
    dists = _site(tmp_path, {
        "numpy": ("2.5.4", None), "PyYAML": ("6.0.3", None), "Jinja2": ("3.1.6", None),
        "torch": ("2.14.1", None), "triton": ("3.8.0", None), "nvidia-cublas": ("13.1.1.3", None),
        "nvidia_cudnn_cu13": ("9.24.0.43", None), "cuda-bindings": ("13.4.4", None), "cuda_toolkit": ("13.0.3.0", None),
        "pytorch-triton-rocm": ("3.1.0", None),
        "colosseum": ("0.1.0", {"url": "file:///repo", "dir_info": {"editable": True}}),
        "mylib": ("1.0", {"url": "file:///src/mylib", "dir_info": {"editable": True}}),
        "localwheel": ("2.0", {"url": "file:///tmp/localwheel-2.0.whl", "archive_info": {}}),
        "fromgit": ("0.3", {"url": "https://github.com/x/fromgit", "vcs_info": {"vcs": "git"}}),
        "torchvision": ("0.29.1+cu130", None),
    })
    assert hub_freeze(dists) == ["fromgit==0.3", "jinja2==3.1.6", "numpy==2.5.4", "pyyaml==6.0.3",
                                 "torchvision==0.29.1"]
    assert EXCLUDED == ("torch", "nvidia-*", "triton", "colosseum", "cuda-*", "pytorch-triton*")


def test_first_distribution_of_a_name_wins(tmp_path):
    a = _site(tmp_path / "a", {"numpy": ("2.5.4", None)})
    b = _site(tmp_path / "b", {"numpy": ("1.26.0", None), "torch": ("2.0", None)})
    assert hub_freeze(a + b) == ["numpy==2.5.4"]


def test_override_file(tmp_path, caplog):
    path = tmp_path / "req.txt"
    path.write_text("# mine\nnumpy>=2  # any\n\ntorch==2.0\nnvidia-cublas\nlz4==4.4.5\t# tab\n"
                    "pkg @ https://example.org/pkg.zip#sha256=00\n")
    with caplog.at_level(logging.WARNING):
        reqs, digest = node_requirements(override=path)
    assert reqs == ["numpy>=2", "lz4==4.4.5", "pkg @ https://example.org/pkg.zip#sha256=00"]
    assert digest == requirements_hash(reqs) and len(digest) == 64
    assert "torch==2.0" in caplog.text and "nvidia-cublas" in caplog.text
    path.write_text("-r other.txt\n")
    with pytest.raises(ValueError, match="pip options"):
        read_requirements_file(path)
    path.write_text("<<<\n")
    with pytest.raises(ValueError, match="not a requirement"):
        read_requirements_file(path)


def test_hash_follows_the_list():
    assert requirements_hash(["a==1"]) != requirements_hash(["a==2"])
    assert requirements_hash(["a==1", "b==2"]) != requirements_hash(["b==2", "a==1"])
    assert canonical_name("Jinja2") == "jinja2" and canonical_name("typing_extensions") == "typing-extensions"


def test_real_environment_list_has_no_hardware_packages():
    reqs, _ = node_requirements(override=None)
    names = [r.split("==")[0] for r in reqs]
    assert "numpy" in names and "grpcio" in names
    assert not any(n == "torch" or n.startswith(("nvidia-", "cuda-")) or n in ("triton", "colosseum") for n in names)
    assert all("+" not in r for r in reqs)
