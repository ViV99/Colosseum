"""Guarantees the test infrastructure (tests/conftest.py) gives every test."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def example_config(name: str) -> Path:
    return REPO_ROOT / "configs" / "examples" / name


def test_spawn_start_method():
    assert mp.get_start_method() == "spawn"


def test_cwd_and_tempfile_are_inside_tmp_path(tmp_path):
    assert Path.cwd() == tmp_path
    with tempfile.TemporaryDirectory() as d:
        assert Path(d).parent == tmp_path


def test_children_use_one_openmp_thread():
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert torch.get_num_threads() == 1


def test_example_configs_resolve_from_any_cwd():
    assert example_config("tic_tac_toe.yaml").is_file()
    assert (REPO_ROOT / "src" / "colosseum").is_dir()


def test_markers_are_registered(pytestconfig):
    markers = "\n".join(pytestconfig.getini("markers"))
    assert "gpu:" in markers and "slow:" in markers


def test_clashing_module_names_fail_the_session_early(tmp_path):
    """Two modules with one basename in different test dirs abort the run with a clear error.

    Test dirs have no ``__init__.py`` and use the default ("prepend") import mode,
    so modules are imported by basename; a clash would otherwise surface as
    "import file mismatch" or as one helper module silently shadowing another.
    """
    tests_dir = tmp_path / "tests"
    for sub in ("unit", "integration"):
        (tests_dir / sub).mkdir(parents=True)
        (tests_dir / sub / "test_same.py").write_text("def test_ok():\n    pass\n")
    shutil.copy(REPO_ROOT / "tests" / "conftest.py", tests_dir / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(tests_dir)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )

    assert proc.returncode == pytest.ExitCode.USAGE_ERROR, proc.stdout + proc.stderr
    output = proc.stdout + proc.stderr
    assert "test_same" in output
    assert str(Path("unit") / "test_same.py") in output
    assert str(Path("integration") / "test_same.py") in output
