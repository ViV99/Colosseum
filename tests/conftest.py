"""Shared pytest configuration for the Colosseum test suite."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

# Must happen before torch is imported anywhere: spawned children inherit the
# environment, so every worker/learner a test starts uses one OpenMP thread
# instead of all cores (tests run several processes on a shared machine).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# `helpers` (tests/) and `examples.*` (repo root) must be importable from every
# test module and every spawned child (spawn copies sys.path into the child).
for _path in (str(REPO_ROOT), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Forking a process that already runs torch/gRPC threads deadlocks (R6-01).
# Production uses spawn too, so tests exercise the same start method.
mp.set_start_method("spawn", force=True)

# Test models are tiny: intra-op threads only add contention with the child
# processes some tests start. Children are fresh interpreters (unaffected).
torch.set_num_threads(1)

# Top-level entries that may legitimately appear in the repo root during a run.
_ALLOWED_NEW_ROOT_ENTRIES = {".pytest_cache", ".ruff_cache", "__pycache__", ".coverage"}


def pytest_configure(config: pytest.Config) -> None:
    """Abort early if two modules under tests/ share a basename.

    Test dirs have no ``__init__.py`` and pytest uses the default "prepend"
    import mode, so every module there (tests and helpers such as ``harness``)
    is imported by its bare basename. Unique basenames keep ``from helpers
    import ...`` style imports and spawn pickling of test-module functions
    working; a clash would otherwise surface as "import file mismatch" or as
    one helper module silently shadowing another.
    """
    by_name: dict[str, list[Path]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.name != "conftest.py":
            by_name.setdefault(path.stem, []).append(path.relative_to(TESTS_DIR))
    clashes = {name: paths for name, paths in by_name.items() if len(paths) > 1}
    if clashes:
        lines = [f"  {name}: {', '.join(map(str, paths))}" for name, paths in sorted(clashes.items())]
        raise pytest.UsageError(
            "module basenames under tests/ must be unique (rename one of each group):\n" + "\n".join(lines)
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``gpu`` when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires a CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _isolate_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test with cwd = tmp_path and tempfile rooted in tmp_path.

    Anything a test (or the code under test, e.g. a default ``./checkpoints``)
    writes to a relative path or a TemporaryDirectory lands under tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


@pytest.fixture(autouse=True, scope="session")
def _repo_root_stays_clean():
    """Fail the session if any test created a new top-level entry in the repo root."""
    before = {p.name for p in REPO_ROOT.iterdir()}
    yield
    created = {p.name for p in REPO_ROOT.iterdir()} - before - _ALLOWED_NEW_ROOT_ENTRIES
    assert not created, f"tests wrote into the repo root: {sorted(created)}"


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Directory for run folders / checkpoints created by a test."""
    root = tmp_path / "runs"
    root.mkdir()
    return root
