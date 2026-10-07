# SP1 Plan — Part A: Tooling (block 0) and Model protocol (block 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Read `00-overview.md` first.** It holds the global constraints, the file map and the binding interface contract that this part implements.

**Scope.** Block 0 gives the project a reproducible dev environment (`scripts/setup-dev.sh`, pinned extras), a test suite that runs in one process without hanging (spawn, markers, timeouts, `tmp_path`-only files, new `tests/unit|contract|integration` layout), ruff + GitHub Actions CI, a throughput baseline measured *before* any performance work, and an in-process testable `RolloutLoop` extracted from the worker as a pure refactor. Block 1 replaces `ActorCriticNetwork` and every `is_recurrent`/`lstm_hidden` branch with the `PolicyModel` protocol: an opaque `State` pytree, `act()`, four cores (none / LSTM / GRU / window attention), `ComposedModel`, the new `networks` config schema with `build_model`/`validate_config`, APPO training through `model.unroll`, a worker that keeps per-slot state, and a contract test proving that the learner reproduces the worker's log-probs and values for all four cores.

**Conventions used in this part**

- Commands run from the repository root. Python is always `.venv/bin/python` (created in T0.1); ruff is `.venv/bin/ruff`.
- "Replace A with B" means an exact, unique string replacement in the named file. Every "A" snippet in this part was checked to occur exactly once in the file as it is at the start of that task (after all earlier tasks of this part).
- From T0.3 on, every task ends with `.venv/bin/ruff check --fix src tests examples scripts` followed by `.venv/bin/ruff check src tests examples scripts` (must print `All checks passed!`). The `--fix` run only removes imports that a task made unused and re-sorts import blocks.
- The fast suite is `.venv/bin/python -m pytest -m "not gpu and not slow" -q`. In this part it takes about 15 s on the 8-core dev machine.
- Until T2.2 (numpy payloads) the process-level pipeline tests are marked `slow` because of the known R6-02 shutdown race (torch tensors in `mp.Queue`). Tasks that change the training path run them explicitly with `-m slow`.

---

### Task T0.1: Dev environment script, dependency pins and extras, optional wandb import

Findings: R3-21 (generated gRPC code needs grpcio ≥ 1.78 / protobuf ≥ 6.31), R3-22 (torch ≥ 2.6), R6-12 (Box2D undeclared), R5 (wandb mandatory).

**Files:**
- Create: `scripts/setup-dev.sh`
- Create: `tests/unit/test_wandb_logger.py`
- Modify: `pyproject.toml` (`dependencies`, `[project.optional-dependencies]`)
- Modify: `src/colosseum/metrics/wandb_logger.py` (`WandBLogger.__init__`, lines 21-33)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `scripts/setup-dev.sh [--gpu]` — idempotent; creates `.venv` (Python 3.12) with CPU torch (or the default-index build with `--gpu`) and `colosseum` installed editable with extras `grpc,dev,examples`.
  - `pyproject.toml` extras: `grpc` (`grpcio>=1.78`, `grpcio-tools>=1.78`, `protobuf>=6.31`), `wandb` (`wandb>=0.16.0`), `examples` (`Box2D>=2.3.10`), `dev` (`pytest>=8.0`, `pytest-timeout>=2.3`, `ruff>=0.6`); core dependency `torch>=2.6`; `wandb` is no longer a core dependency.
  - `WandBLogger(config, run_name=None)` never raises when `wandb` is missing or fails to initialize; it logs a warning naming the `wandb` extra and becomes a no-op.

- [ ] **Step 1: Create the working branch**

```bash
git switch sp1-stabilization 2>/dev/null || git switch -c sp1-stabilization
```

- [ ] **Step 2: Write `scripts/setup-dev.sh`**

Create `scripts/setup-dev.sh` with exactly this content and make it executable (`chmod +x scripts/setup-dev.sh`). Only the CPU index is passed for CPU torch: uv gives extra indexes priority, so `--extra-index-url` would pull the CUDA build. Torch is reinstalled only when the installed build (CPU vs default) differs from the requested one, so re-running the script is cheap.

```bash
#!/usr/bin/env bash
# Create or refresh the development environment in ./.venv (idempotent).
#
#   scripts/setup-dev.sh          CPU-only torch from the PyTorch CPU index (default)
#   scripts/setup-dev.sh --gpu    torch from the default PyPI index (CUDA build on Linux)
#
# Installs uv into ~/.local/bin if it is missing, creates .venv with Python 3.12,
# installs torch, then the project in editable mode with the grpc, dev and
# examples extras. Never touches the system Python.
set -euo pipefail

GPU=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "setup-dev.sh: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo ">>> installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi

if [ ! -x .venv/bin/python ]; then
  echo ">>> creating .venv (Python 3.12)"
  uv venv .venv --python 3.12
fi
PY=.venv/bin/python

# Reinstall torch only when the wanted build (CPU vs default) is not installed yet.
CURRENT="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
NEED_TORCH=0
if [ -z "$CURRENT" ]; then
  NEED_TORCH=1
elif [ "$GPU" -eq 0 ] && [[ "$CURRENT" != *+cpu ]]; then
  NEED_TORCH=1
elif [ "$GPU" -eq 1 ] && [[ "$CURRENT" == *+cpu ]]; then
  NEED_TORCH=1
fi
if [ "$NEED_TORCH" -eq 1 ]; then
  if [ "$GPU" -eq 1 ]; then
    echo ">>> installing torch (default index)"
    uv pip install --python "$PY" --reinstall-package torch "torch>=2.6"
  else
    # Only the CPU index: uv prefers extra indexes, so --extra-index-url would pull CUDA wheels.
    echo ">>> installing CPU torch from https://download.pytorch.org/whl/cpu"
    uv pip install --python "$PY" --reinstall-package torch \
      --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"
  fi
else
  echo ">>> torch $CURRENT already installed"
fi

echo ">>> installing colosseum (editable) with extras grpc, dev, examples"
uv pip install --python "$PY" -e ".[grpc,dev,examples]"

"$PY" - <<'PYEOF'
import torch
print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
PYEOF
echo ">>> done. Run tests with: .venv/bin/python -m pytest -m 'not gpu and not slow' -q"
```

- [ ] **Step 3: Update the dependency pins in `pyproject.toml`**

In `pyproject.toml` replace:

```toml
dependencies = [
    "torch>=2.2.0",
    "numpy>=1.26.0",
    "gymnasium>=1.0.0",
    "pyyaml>=6.0",
    "pydantic>=2.5.0",
    "lz4>=4.3.0",
    "wandb>=0.16.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
grpc = [
    "grpcio>=1.60.0",
    "grpcio-tools>=1.60.0",
    "protobuf>=4.25.0",
]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.1.0",
]
```

with:

```toml
dependencies = [
    "torch>=2.6",
    "numpy>=1.26.0",
    "gymnasium>=1.0.0",
    "pyyaml>=6.0",
    "pydantic>=2.5.0",
    "lz4>=4.3.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
# The generated code in colosseum/transport/colosseum_pb2*.py requires these versions.
grpc = [
    "grpcio>=1.78",
    "grpcio-tools>=1.78",
    "protobuf>=6.31",
]
wandb = [
    "wandb>=0.16.0",
]
examples = [
    "Box2D>=2.3.10",  # examples/space_miners (import Box2D); has cp312 manylinux wheels
]
dev = [
    "pytest>=8.0",
    "pytest-timeout>=2.3",
    "ruff>=0.6",
]
```

`examples/space_miners/game_engine.py` does `import Box2D`; the PyPI distribution providing that module with Python 3.12 wheels is `Box2D` (2.3.10). `box2d-py` has no 3.12 wheels.

- [ ] **Step 4: Create the environment**

Run: `bash scripts/setup-dev.sh`

Expected: the last lines are `torch 2.x.y+cpu  cuda available: False` and `>>> done. ...`. Running it a second time prints `>>> torch 2.x.y+cpu already installed` and finishes quickly. Check the extras: `.venv/bin/python -c "import grpc, google.protobuf, Box2D, pytest_timeout; print(grpc.__version__, google.protobuf.__version__)"` prints versions ≥ 1.78 / ≥ 6.31.

- [ ] **Step 5: Write the failing test for the optional wandb import**

Create `tests/unit/test_wandb_logger.py`:

```python
"""WandB is an optional dependency: missing package must only disable logging."""

from __future__ import annotations

import logging
import subprocess
import sys

from colosseum.core.config import MetricsConfig
from colosseum.metrics.wandb_logger import WandBLogger


def test_missing_wandb_package_disables_logging(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise ImportError
    with caplog.at_level(logging.WARNING, logger="colosseum.metrics.wandb_logger"):
        wb = WandBLogger(MetricsConfig(use_wandb=True))
    assert wb._enabled is False
    assert "uv pip install -e '.[wandb]'" in caplog.text
    wb.log_config({"a": 1})
    wb.log_train_step("agent_0", {"loss": 1.0}, step=1)
    wb.finish()


def test_disabled_logger_never_imports_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    wb = WandBLogger(MetricsConfig(use_wandb=False))
    wb.log_train_step("agent_0", {"loss": 1.0}, step=1)
    wb.finish()


def test_package_imports_without_wandb_installed():
    code = (
        "import sys; sys.modules['wandb'] = None; "
        "import colosseum.cli, colosseum.launcher, colosseum.metrics.wandb_logger"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 6: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/unit/test_wandb_logger.py -v`

Expected: `test_missing_wandb_package_disables_logging` FAILS on `assert "uv pip install -e '.[wandb]'" in caplog.text` (the current message is `Failed to initialize WandB: ...`). The other two tests pass (wandb is already imported lazily); they guard against regressions.

- [ ] **Step 7: Make the import failure explicit**

In `src/colosseum/metrics/wandb_logger.py` replace:

```python
        if self._enabled:
            try:
                import wandb

                self._run = wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=run_name,
                    config={},  # will be updated with full config
                )
            except (ImportError, RuntimeError, OSError) as e:
                logger.warning(f"Failed to initialize WandB: {e}. Logging disabled.")
                self._enabled = False
```

with:

```python
        if self._enabled:
            try:
                import wandb
            except ImportError:
                logger.warning(
                    "metrics.use_wandb is true but the 'wandb' package is not installed "
                    "(it is an optional extra: uv pip install -e '.[wandb]'). WandB logging disabled."
                )
                self._enabled = False
            else:
                try:
                    self._run = wandb.init(
                        project=config.wandb_project,
                        entity=config.wandb_entity,
                        name=run_name,
                        config={},  # will be updated with full config
                    )
                except Exception as e:  # any wandb init failure must not stop training
                    logger.warning(f"Failed to initialize WandB: {e}. Logging disabled.")
                    self._enabled = False
```

- [ ] **Step 8: Run the test again**

Run: `.venv/bin/python -m pytest tests/unit/test_wandb_logger.py -v`

Expected: 3 passed.

- [ ] **Step 9: Run the suite that is safe before T0.2**

The full suite still deadlocks under the default `fork` start method (R6-01); T0.2 fixes that. Until then run everything except the tests that start a `Launcher` or a worker process:

```bash
.venv/bin/python -m pytest tests -q --ignore=tests/test_integration.py \
  -k "not full_pipeline and not multi_agent_pipeline and not worker_multi_agent_routing"
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit**

```bash
git add scripts/setup-dev.sh pyproject.toml src/colosseum/metrics/wandb_logger.py tests/unit/test_wandb_logger.py
git commit -m "chore: add setup-dev.sh, pin torch/grpc versions, make wandb and Box2D optional extras"
```

---

### Task T0.2: Test infrastructure: spawn, markers, timeouts, tmp_path-only, new test layout

Findings: R6-01 (fork deadlock: the full suite hung at `test_integration.py::test_full_pipeline`), R4-21 (tests write `./checkpoints`), R6-04 (thread oversubscription made pipeline tests take 2-3 min), `run_*_test.py` scripts that print "completed" even when a child crashed.

**Files:**
- Create: `tests/unit/test_infrastructure.py`, `tests/integration/test_pipelines.py`, `tests/integration/test_distributed_e2e.py`
- Modify: `tests/conftest.py` (rewrite), `tests/helpers.py` (header), `pyproject.toml` (`[tool.pytest.ini_options]`)
- Move (`git mv`): `tests/test_{action_masking,appo,bc,composite_actions,config,distributions,eval,milestone2,multi_agent,performance,ratings,recurrent,review_fixes,vec_env,vtrace}.py` → `tests/unit/`; `tests/test_{grpc,distributed,subproc_vec_env}.py` → `tests/integration/`
- Delete: `tests/test_integration.py` (its tests move to `tests/integration/test_pipelines.py`), `tests/run_pipeline_test.py`, `tests/run_scaled_test.py`, `tests/run_worker_test.py`, `tests/run_subproc_pipeline_test.py`, `tests/run_distributed_test.py`
- Modify after the move: `tests/unit/test_multi_agent.py`, `tests/unit/test_milestone2.py`, `tests/unit/test_performance.py`, `tests/unit/test_recurrent.py`, `tests/unit/test_review_fixes.py`, `tests/unit/test_ratings.py`, `tests/unit/test_config.py`, `tests/integration/test_distributed.py`, `tests/integration/test_subproc_vec_env.py`

**Interfaces:**
- Consumes: T0.1 (`pytest-timeout` in the `dev` extra, `.venv`).
- Produces:
  - Every test runs with `cwd == tmp_path` and `tempfile.tempdir == tmp_path` (autouse fixture); the session fails if a test creates a new top-level entry in the repo root.
  - Start method `spawn` for the whole session; `torch.set_num_threads(1)` in the pytest process; `OMP_NUM_THREADS=1` inherited by every spawned child.
  - Markers `gpu` (auto-skipped without CUDA) and `slow`; `--strict-markers`; default pytest-timeout 300 s.
  - Fixture `run_root(tmp_path) -> Path` (`tmp_path / "runs"`, created).
  - `tests/helpers.py`: `REPO_ROOT: Path`, `example_config(name: str) -> Path`. `helpers` and `examples.*` are importable from every test module and every spawned child.
  - Layout `tests/unit/`, `tests/integration/` (`tests/contract/` is created in T0.5, `tests/learning/` in T8.2). Test file basenames stay unique across directories (no `__init__.py` files).

- [ ] **Step 1: Write the failing infrastructure test**

Create `tests/unit/test_infrastructure.py`:

```python
"""Guarantees the test infrastructure (tests/conftest.py) gives every test."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import torch

from helpers import REPO_ROOT, example_config


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
```

- [ ] **Step 2: Run it and see it fail**

Run: `.venv/bin/python -m pytest tests/unit/test_infrastructure.py -v`

Expected: collection error `ImportError: cannot import name 'REPO_ROOT' from 'helpers'` (or `ModuleNotFoundError: No module named 'helpers'`).

- [ ] **Step 3: Rewrite `tests/conftest.py`**

Replace the whole file with:

```python
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
```

- [ ] **Step 4: Add the path helpers to `tests/helpers.py`**

In `tests/helpers.py` replace:

```python
"""Shared test helpers: simple network components for unit tests."""

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist
```

with:

```python
"""Shared test helpers: toy environments and small networks/models."""

from pathlib import Path

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist

REPO_ROOT = Path(__file__).resolve().parent.parent


def example_config(name: str) -> Path:
    """Absolute path of ``configs/examples/<name>`` (tests run with cwd = tmp_path)."""
    return REPO_ROOT / "configs" / "examples" / name
```

- [ ] **Step 5: Register markers and the default timeout in `pyproject.toml`**

In `pyproject.toml` replace:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
timeout = 300
markers = [
    "gpu: needs a CUDA device (auto-skipped without CUDA)",
    "slow: longer than ~20 s, or races known until T2.2; excluded from the fast suite (-m 'not gpu and not slow')",
]
```

- [ ] **Step 6: Run the infrastructure test**

Run: `.venv/bin/python -m pytest tests/unit/test_infrastructure.py -v`

Expected: 5 passed.

- [ ] **Step 7: Move the test files into the new layout and delete the ad-hoc scripts**

```bash
mkdir -p tests/unit tests/integration
for f in action_masking appo bc composite_actions config distributions eval milestone2 multi_agent \
         performance ratings recurrent review_fixes vec_env vtrace; do
  git mv tests/test_$f.py tests/unit/test_$f.py
done
for f in grpc distributed subproc_vec_env; do
  git mv tests/test_$f.py tests/integration/test_$f.py
done
git rm -q tests/test_integration.py tests/run_pipeline_test.py tests/run_scaled_test.py \
  tests/run_worker_test.py tests/run_subproc_pipeline_test.py tests/run_distributed_test.py
```

Why the scripts go: `run_pipeline_test.py`, `run_worker_test.py` and `run_subproc_pipeline_test.py` become assertions in `tests/integration/test_pipelines.py` (Step 12); `run_distributed_test.py` becomes `tests/integration/test_distributed_e2e.py`; `run_scaled_test.py` (4 workers, 50k steps) is a benchmark, superseded by `scripts/bench_throughput.py` (T0.4), and violates the 2-worker limit for tests.

- [ ] **Step 8: Remove per-file `sys.path` hacks and fix imports that depended on the old layout**

`conftest.py` now puts the repo root and `tests/` on `sys.path`, so every `sys.path.insert(0, ...)` line in the moved files is obsolete (they would now point at the wrong directories). Imports they leave unused are removed by ruff in T0.3.

```bash
sed -i '/^sys\.path\.insert(0, /d' tests/unit/*.py tests/integration/*.py
sed -i 's/from tests\.helpers import/from helpers import/' tests/unit/test_review_fixes.py
sed -i 's/"tests\.test_recurrent\.\(Simple[A-Za-z]*\)"/"helpers.\1"/' tests/unit/test_recurrent.py
sed -i 's/    from tests\.test_action_masking import _make_network/    from helpers import make_simple_network/; s/    net = _make_network(obs_dim=4, num_actions=3)/    net = make_simple_network(obs_dim=4, hidden_dim=32, num_actions=3)/' tests/unit/test_performance.py
```

Check: `grep -rn "sys.path\|from tests\.\|tests\.test_" tests/unit tests/integration` prints nothing.

- [ ] **Step 9: Make config paths independent of the working directory**

Tests now run with `cwd = tmp_path`, so relative `configs/examples/...` paths must become absolute:

```bash
for f in tests/unit/test_multi_agent.py tests/unit/test_ratings.py tests/unit/test_milestone2.py tests/unit/test_config.py; do
  sed -i 's#load_config("configs/examples/\([a-z_]*\.yaml\)")#load_config(example_config("\1"))#' "$f"
  sed -i '0,/^import /s//from helpers import example_config\nimport /' "$f"
done
```

The second `sed` inserts `from helpers import example_config` before the first `import` line; ruff sorts it in T0.3. Check: `grep -rn '"configs/examples' tests` prints nothing.

- [ ] **Step 10: Remove the tests that moved to `tests/integration/test_pipelines.py`, and `__main__` blocks**

Save this helper as `/tmp/drop_sections.py` (it is not part of the repo):

```python
import re
import sys

# usage: python /tmp/drop_sections.py FILE [section-title-prefix ...]
# Deletes "# ----\n# <title>\n# ----" sections (header + body up to the next
# header) whose title starts with one of the prefixes, and a trailing
# `if __name__ == "__main__":` block.
path, *titles = sys.argv[1:]
text = open(path).read()
header = re.compile(r"^# -{10,}\n# (?P<title>.*)\n# -{10,}\n", re.M)
marks = list(header.finditer(text))
out, pos = [], 0
for i, m in enumerate(marks):
    end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
    if any(m.group("title").startswith(t) for t in titles):
        out.append(text[pos:m.start()])
        pos = end
out.append(text[pos:])
text = "".join(out)
cut = text.find('\nif __name__ == "__main__":')
if cut != -1:
    text = text[:cut]
open(path, "w").write(text.rstrip() + "\n")
```

Then run:

```bash
.venv/bin/python /tmp/drop_sections.py tests/unit/test_multi_agent.py \
  "T1.4: Worker multi-agent routing" "T1.4d: Full multi-agent pipeline"
for f in tests/unit/test_review_fixes.py tests/integration/test_distributed.py tests/integration/test_subproc_vec_env.py; do
  .venv/bin/python /tmp/drop_sections.py "$f"   # no titles: only strips the __main__ block
done
.venv/bin/python - <<'EOF'
p = "tests/unit/test_milestone2.py"
s = open(p).read()
s = s[: s.index("\ndef test_full_pipeline_with_checkpoints(")]
open(p, "w").write(s.rstrip() + "\n")
EOF
```

Check: `grep -n "def test_" tests/unit/test_multi_agent.py` lists `test_agent_config_defaults`, `test_get_trainable_agent_ids_empty`, `test_get_trainable_agent_ids_multi`, `test_get_agent_config_no_override`, `test_get_agent_config_with_override`, `test_load_multi_agent_config_roundtrip`, `test_derive_worker_configs`, `test_monitor_loop_per_agent_checkpoint_queues`; `grep -n "def test_" tests/unit/test_milestone2.py` lists `test_checkpoint_manager`, `test_matchmaker`, `test_coordinator`, `test_derive_worker_configs`; `grep -rn "__main__" tests` only finds the docstring of `test_subproc_vec_env.py`.

- [ ] **Step 11: Make the `torch.compile` checks lazy and slow**

`_torch_compile_available()` compiles a function at collection time (inside `skipif`). Move it into the tests and mark them `slow` (compiling takes 3-15 s):

In `tests/unit/test_performance.py` replace:

```python
@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile backend unavailable")
def test_vtrace_torch_compile():
    """Compiled V-trace should produce same results as eager."""
```

with:

```python
@pytest.mark.slow
def test_vtrace_torch_compile():
    """Compiled V-trace should produce same results as eager."""
    if not _torch_compile_available():
        pytest.skip("torch.compile backend unavailable")
```

and replace:

```python
@pytest.mark.skipif(not _torch_compile_available(), reason="torch.compile backend unavailable")
def test_appo_torch_compile():
    """APPO should train successfully with use_torch_compile=True."""
```

with:

```python
@pytest.mark.slow
def test_appo_torch_compile():
    """APPO should train successfully with use_torch_compile=True."""
    if not _torch_compile_available():
        pytest.skip("torch.compile backend unavailable")
```

- [ ] **Step 12: Write the process-level pipeline tests**

Create `tests/integration/test_pipelines.py`. It replaces `tests/test_integration.py`, the routing and pipeline tests removed in Step 10 and the `run_*` scripts, with real assertions: chunks of the right agent and length, and saved checkpoints proving the learner trained. Checkpoint intervals are chosen so that the last checkpoint is sent at least ~30 train steps before the learner stops; a checkpoint still queued when the learner exits crashes the main process (R6-02, fixed in T2.2). For the same reason the `Launcher` tests are `slow` until T2.2.

```python
"""End-to-end runs through real worker / learner processes (spawn start method)."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import pytest

from colosseum.coordinator.checkpoint_manager import CheckpointManager
from colosseum.core.config import ColosseumConfig, load_config
from helpers import example_config


def _config(name: str, tmp_path, **sections: dict) -> ColosseumConfig:
    """Example config with checkpoints under tmp_path, WandB off and section overrides."""
    data = load_config(example_config(name)).model_dump()
    data["metrics"]["use_wandb"] = False
    data["checkpoint"]["dir"] = str(tmp_path / "checkpoints")
    for section, values in sections.items():
        data[section].update(values)
    return ColosseumConfig(**data)


def _stop(proc: mp.Process, stop_event) -> None:
    stop_event.set()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)


def test_worker_produces_chunks(tmp_path):
    """A spawned worker process sends full-length chunks for its agent."""
    from colosseum.launcher import _worker_target

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
    )
    agent_id = "agent_0"
    trajectory_queues = {agent_id: mp.Queue(maxsize=16)}
    weight_queues = {agent_id: mp.Queue(maxsize=2)}
    stop_event = mp.Event()
    proc = mp.Process(
        target=_worker_target,
        args=(0, config, [agent_id], {agent_id: config}, trajectory_queues, weight_queues, stop_event, 500),
        daemon=True,
    )
    proc.start()
    try:
        chunks = [trajectory_queues[agent_id].get(timeout=60) for _ in range(5)]
    finally:
        _stop(proc, stop_event)
    for chunk in chunks:
        assert chunk.agent_id == agent_id
        assert chunk.observations.shape[0] == 8


def test_worker_multi_agent_routing(tmp_path):
    """Chunks are routed to the queue of the agent that occupies the slot."""
    from colosseum.launcher import _worker_target

    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
    )
    agent_ids = config.get_trainable_agent_ids()
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_ids}
    trajectory_queues = {aid: mp.Queue(maxsize=16) for aid in agent_ids}
    weight_queues = {aid: mp.Queue(maxsize=2) for aid in agent_ids}
    stop_event = mp.Event()
    slot_agent_map = [[agent_ids[0], agent_ids[1]], [agent_ids[0], agent_ids[1]]]
    slot_network_map = [["latest", "latest"], ["latest", "latest"]]
    collect_mask = [[True, True], [True, True]]
    proc = mp.Process(
        target=_worker_target,
        args=(
            0, config, agent_ids, agent_configs, trajectory_queues, weight_queues,
            stop_event, 500, None, slot_network_map, collect_mask, slot_agent_map, None,
        ),
        daemon=True,
    )
    proc.start()
    chunks_by_agent: dict[str, list] = {aid: [] for aid in agent_ids}
    deadline = time.time() + 60
    try:
        while time.time() < deadline and min(len(v) for v in chunks_by_agent.values()) < 2:
            for aid in agent_ids:
                try:
                    chunks_by_agent[aid].append(trajectory_queues[aid].get(timeout=0.5))
                except queue.Empty:
                    pass
    finally:
        _stop(proc, stop_event)
    for aid in agent_ids:
        assert len(chunks_by_agent[aid]) >= 2, f"agent {aid} received too few chunks"
        assert all(c.agent_id == aid for c in chunks_by_agent[aid])


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_full_pipeline(tmp_path):
    """Single-agent self-play training runs to completion and saves checkpoints."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    Launcher(config).launch()
    assert CheckpointManager(str(tmp_path / "checkpoints")).list_checkpoints("agent_0")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_full_pipeline_with_checkpoint_pool(tmp_path):
    """Checkpoints are saved every N train steps and the FIFO pool is respected."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 5000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        # 312 train steps: checkpoints at 40, 80, ..., 280 -> FIFO keeps the last 5.
        self_play={"checkpoint_interval": 40, "pool_size": 5},
    )
    Launcher(config).launch()
    ckpts = CheckpointManager(str(tmp_path / "checkpoints"), pool_size=5).list_checkpoints("agent_0")
    assert [c.policy_version for c in ckpts] == [120, 160, 200, 240, 280]


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_multi_agent_pipeline(tmp_path):
    """Two-agent league training runs to completion."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe_multi.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
        self_play={"checkpoint_interval": 50},
    )
    Launcher(config).launch()
    manager = CheckpointManager(str(tmp_path / "checkpoints"))
    assert any(manager.list_checkpoints(aid) for aid in config.get_trainable_agent_ids())


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_subprocess_vec_env_pipeline(tmp_path):
    """Nested spawn (worker -> env subprocesses) trains to completion."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 2000},
        rollout={
            "num_workers": 1, "envs_per_worker": 4, "chunk_length": 8,
            "vec_env": "subprocess", "subproc_workers": 2, "match_refresh_interval_sec": 1.0,
        },
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    Launcher(config).launch()
    assert CheckpointManager(str(tmp_path / "checkpoints")).list_checkpoints("agent_0")
```

Create `tests/integration/test_distributed_e2e.py` (from `run_distributed_test.py`, with checkpoints under `tmp_path`):

```python
"""End-to-end distributed (gRPC) run on localhost: weight store + learner + workers."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time

import pytest

from helpers import example_config

pytest.importorskip("grpc")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_distributed_grpc_pipeline(tmp_path):
    """The learner trains on chunks from gRPC workers and publishes weights (version > 0)."""
    from colosseum.distributed import run_distributed_learner, run_distributed_workers
    from colosseum.weight_store.grpc_store import GRPCWeightStore, serve_weight_store

    ws_port, traj_port = _free_port(), _free_port()
    ws_addr, learner_addr = f"localhost:{ws_port}", f"localhost:{traj_port}"
    agent = "agent_0"
    cfg_path = str(example_config("tic_tac_toe.yaml"))
    overrides = {
        "training.total_timesteps": 1200,
        "rollout.num_workers": 2,
        "rollout.envs_per_worker": 4,
        "rollout.chunk_length": 8,
        "rollout.weight_sync_interval_sec": 0.5,
        "learner.batch_chunks": 2,
        "learner.queue_size": 32,
        "self_play.checkpoint_interval": 1,
        "metrics.use_wandb": False,
        "checkpoint.dir": str(tmp_path / "checkpoints"),
    }

    ws_server = serve_weight_store(port=ws_port)
    learner = mp.Process(
        target=run_distributed_learner,
        args=(cfg_path, agent, traj_port, ws_addr, overrides),
        daemon=False,
    )
    learner.start()
    try:
        time.sleep(2.0)  # let the TrajectoryService bind
        run_distributed_workers(cfg_path, ws_addr, {agent: learner_addr}, overrides)
        time.sleep(2.0)  # let the learner drain the last chunks
    finally:
        learner.terminate()
        learner.join(timeout=10)

    client = GRPCWeightStore(ws_addr)
    try:
        version = client.get_version(agent)
        payload = client.get(agent)
    finally:
        client.close()
        ws_server.stop(0)
    assert payload is not None, "no weights were published to the store"
    assert version > 0, f"learner did not train/publish (version={version})"
```

- [ ] **Step 13: Run the fast suite and check durations**

```bash
rm -rf checkpoints   # left in the repo root by earlier runs of the old suite
.venv/bin/python -m pytest -m "not gpu and not slow" -q --durations=15
```

Expected: all tests pass in well under a minute (about 15 s on the 8-core dev machine; `tests/unit/test_eval.py::test_evaluate_three_agents` went from 42 s to well under 1 s with one intra-op thread). No test in the durations list takes more than 20 s; if one does on your machine, add `@pytest.mark.slow` to it and note it in the commit message.

- [ ] **Step 14: Verify the full suite (including slow) no longer hangs**

Run: `.venv/bin/python -m pytest -m "not gpu" -q --durations=10`

Expected: all tests pass in about 1.5-2 minutes; the slow ones are the four `Launcher` pipelines, the gRPC end-to-end run (~10 s each with one OpenMP thread per process) and the two `torch.compile` tests. No hang: every test is bounded by pytest-timeout. Worker processes may print tracebacks ending in `FileNotFoundError` from `torch/multiprocessing/reductions.py` at shutdown; that is R6-02 (a worker reading weights from a learner that already exited) and does not fail the test.

Then check that nothing was written to the repo root: `git status --porcelain` shows only the intended moves/edits (no `checkpoints/`, no stray files).

- [ ] **Step 15: Commit**

```bash
git add -A tests pyproject.toml
git commit -m "test: spawn start method, markers, timeouts, tmp_path-only files and unit/integration layout"
```

---

### Task T0.3: Ruff config + GitHub Actions CI

**Files:**
- Modify: `pyproject.toml` (`[tool.ruff]`, new `[tool.ruff.lint]`, `[tool.ruff.lint.isort]`)
- Create: `.github/workflows/ci.yml`
- Modify (autofix + manual fixes): files under `src/`, `tests/`, `examples/` listed in Step 4

**Interfaces:**
- Consumes: T0.1 (`scripts/setup-dev.sh`, `ruff` in `dev`), T0.2 (layout, markers).
- Produces:
  - Lint rules `E, F, I, B, UP` (ignoring `B027`, `B905`, `UP042`), line length 120, target py311, generated `colosseum_pb2*.py` and `review/`, `research/` excluded; `helpers` and `harness` count as first-party for import sorting.
  - `.venv/bin/ruff check src tests examples scripts` passes; every later task keeps it passing (see "Conventions").
  - CI workflow `ci` on push to `main`/`sp1-stabilization` and on pull requests: `scripts/setup-dev.sh` (Python 3.12, CPU torch), ruff, fast suite.

- [ ] **Step 1: Add the ruff configuration**

In `pyproject.toml` replace:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"
```

with:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"
extend-exclude = [
    "src/colosseum/transport/colosseum_pb2.py",
    "src/colosseum/transport/colosseum_pb2_grpc.py",
    "review",
    "research",
]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = [
    "B027",   # empty non-abstract methods in ABCs are intentional optional hooks
    "B905",   # zip() without strict= : lengths are checked explicitly where it matters
    "UP042",  # keep `class X(str, Enum)`; StrEnum changes str() formatting
]

[tool.ruff.lint.isort]
known-first-party = ["colosseum", "examples", "harness", "helpers"]
```

- [ ] **Step 2: See the current violations**

Run: `.venv/bin/ruff check src tests examples --statistics`

Expected: roughly 220 violations (the exact count depends on the ruff version), almost all auto-fixable (`UP045` `Optional[X]` → `X | None`, `I001` import order, `F401` unused imports — many of them left by the `sys.path` removal in T0.2, `UP035`, `UP037`, `UP004`, `UP009`).

- [ ] **Step 3: Apply the safe autofixes**

Run: `.venv/bin/ruff check --fix src tests examples`

Expected: `Found ~250 errors (~235 fixed, 17 remaining)` (the fixer counts iteratively, so the number is higher than in Step 2). The fixes are mechanical (annotation syntax, import order, unused imports). Pydantic models keep working with `X | None` on Python 3.11+.

- [ ] **Step 4: Fix the remaining violations by hand**

1. `examples/composite_action/env.py` (E501; both branches of the conditional were identical) — replace:

```python
            s = float(np.clip(a["speed"], 0.0, 1.0)) if not isinstance(a["speed"], (int, float)) else float(np.clip(a["speed"], 0.0, 1.0))
```

   with:

```python
            s = float(np.clip(a["speed"], 0.0, 1.0))
```

2. `src/colosseum/cli.py` (E501) — replace:

```python
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Path to BC data (.pt file or directory)")
```

   with:

```python
@click.option(
    "--data", "-d", required=True, type=click.Path(exists=True), help="Path to BC data (.pt file or directory)",
)
```

3. `src/colosseum/cli.py` (E501) — replace:

```python
@click.option("--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode) instead of sampling")
```

   with:

```python
@click.option(
    "--deterministic", is_flag=True, default=False, help="Act greedily (distribution mode) instead of sampling",
)
```

4. `src/colosseum/learner/learner.py` (F841 (`device` was never used)) — replace:

```python
    # Create algorithm and network
    device = _resolve_device(config.device)
    algorithm = algorithm_factory()
```

   with:

```python
    # Create algorithm and network
    algorithm = algorithm_factory()
```

5. `src/colosseum/networks/base.py` (F821 (`Distribution` in the `BasePolicy.forward` annotation was undefined)) — replace:

```python
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
```

   with:

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from colosseum.networks.distributions import Distribution
```

6. `src/colosseum/worker/rollout_worker.py` (B007) — replace:

```python
    for env_idx in range(num_envs):
        env_buffers = []
        for p in range(num_players):
            env_buffers.append(RolloutBuffer(
```

   with:

```python
    for _ in range(num_envs):
        env_buffers = []
        for _ in range(num_players):
            env_buffers.append(RolloutBuffer(
```

7. `tests/integration/test_distributed.py` (B011) — replace:

```python
            assert False, "expected queue.Empty before any weights"
```

   with:

```python
            raise AssertionError("expected queue.Empty before any weights")
```

8. `tests/integration/test_distributed.py` (B011) — replace:

```python
            assert False, "expected Empty when version unchanged"
```

   with:

```python
            raise AssertionError("expected Empty when version unchanged")
```

9. `tests/unit/test_appo.py` (E741) — replace:

```python
    assert not all(l == losses[0] for l in losses)
```

   with:

```python
    assert not all(loss == losses[0] for loss in losses)
```

10. `tests/unit/test_milestone2.py` (E402 (pytest captures logs; the module-level basicConfig sat between imports)) — delete:

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
```

11. `tests/unit/test_multi_agent.py` (F841) — delete:

```python
    # Create a minimal launcher and manually invoke checkpoint processing
    launcher = Launcher(config)
```

12. `tests/unit/test_recurrent.py` (B007) — replace:

```python
    for i in range(num_chunks):
```

   with:

```python
    for _ in range(num_chunks):
```

13. `tests/unit/test_recurrent.py` (E741) — replace:

```python
    assert all(np.isfinite(l) for l in losses)
```

   with:

```python
    assert all(np.isfinite(loss) for loss in losses)
```

`B011` also flags three `assert False, "Should have raised ValueError"` lines in `tests/unit/test_composite_actions.py`; replace each of them with `raise AssertionError("Should have raised ValueError")`:

```bash
sed -i 's/        assert False, "Should have raised ValueError"/        raise AssertionError("Should have raised ValueError")/' tests/unit/test_composite_actions.py
```

- [ ] **Step 5: Re-run the autofix (imports made unused by Step 4) and check**

```bash
.venv/bin/ruff check --fix src tests examples scripts
.venv/bin/ruff check src tests examples scripts
```

Expected: `All checks passed!`

- [ ] **Step 6: Add the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main, sp1-stabilization]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - name: Set up environment (Python 3.12, CPU torch, extras grpc/dev/examples)
        run: bash scripts/setup-dev.sh
      - name: Ruff
        run: .venv/bin/ruff check src tests examples scripts
      - name: Tests (no gpu, no slow)
        run: .venv/bin/python -m pytest -m "not gpu and not slow" -q
```

The workflow reuses `scripts/setup-dev.sh`, so CI also exercises the dev setup (CPU torch from the PyTorch CPU index only). `astral-sh/setup-uv` puts `uv` on `PATH`, so the script skips its own uv installation.

- [ ] **Step 7: Run the fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q`

Expected: all pass (the autofixes are behavior-preserving).

- [ ] **Step 8: Commit**

```bash
git add -A pyproject.toml .github src tests examples
git commit -m "chore: ruff config (E,F,I,B,UP) with autofixes and GitHub Actions CI"
```

---

### Task T0.4: Throughput benchmark + `docs/benchmarks.md` baseline

The baseline must be measured **before** any thread or data-flow fix (T2.x), on the same machine that will produce the "after" numbers in T8.3. Run this task before starting block 2 (it only depends on T0.1; doing it right after T0.3 is the intended order). The code changes of T0.1-T0.3 do not affect throughput (optional wandb import, test infrastructure, mechanical lint fixes), so this still is the spec's "before any code change" baseline.

**Files:**
- Create: `scripts/bench_throughput.py`
- Create: `docs/benchmarks.md` (Russian)

**Interfaces:**
- Consumes: T0.1 (`.venv`), `colosseum.launcher.Launcher`, `colosseum.launcher.WandBLogger` (patched at runtime), learner metrics keys `train_step` and `chunks_received`.
- Produces:
  - `scripts/bench_throughput.py [--workers N ...] [--duration S] [--warmup S] [--json PATH]` — prints one line per worker count, a markdown table `| workers | updates/s | env steps/s |` and whether env steps/s grows monotonically; `--json` writes `{"machine": {...}, "results": [{"workers", "updates_per_s", "env_steps_per_s", "train_steps"}]}`.
  - `docs/benchmarks.md` with the section «До» (measured) and an empty «После» section for T8.3.
  - Note for T6.3/T6.4/T8.3: the script reads learner metrics by replacing `colosseum.launcher.WandBLogger`; if the launcher stops routing learner metrics through that class, switch `_Recorder` to the new source (e.g. `metrics.jsonl`) without changing the measured quantities.

- [ ] **Step 1: Write the script**

Create `scripts/bench_throughput.py`:

```python
"""Throughput benchmark: tic-tac-toe self-play training for a fixed wall-clock time.

For every worker count the real ``Launcher`` runs (spawned worker and learner
processes) with the tic-tac-toe example config. The learner's metrics are
captured in this process by replacing ``colosseum.launcher.WandBLogger`` with
an in-memory recorder (the monitor loop forwards every learner metrics dict to
it). After a warm-up period the script measures, over ``--duration`` seconds:

- ``updates/s``   learner train steps per second;
- ``env_steps/s`` env steps consumed by the learner per second, i.e.
  ``chunks_received * chunk_length / num_players`` per second. Checkpoints are
  disabled, so every slot plays the latest policy and collects data, and a
  full queue blocks the workers, so consumption equals production.

Usage::

    .venv/bin/python scripts/bench_throughput.py                 # 1, 2, 4 workers, 60 s each
    .venv/bin/python scripts/bench_throughput.py --workers 1 2 --duration 30 --json out.json

When the launcher stops reporting learner metrics through ``WandBLogger``
(metrics.jsonl, block 6), switch ``_Recorder`` to that source.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import platform
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # `examples.*` for this process and spawned children

CONFIG = REPO_ROOT / "configs" / "examples" / "tic_tac_toe.yaml"
ENVS_PER_WORKER = 8
CHUNK_LENGTH = 32
BATCH_CHUNKS = 8

# (monotonic time, train_step, chunks_received) per learner metrics message.
_SAMPLES: list[tuple[float, int, int]] = []


class _Recorder:
    """Stand-in for WandBLogger with the same interface; records learner metrics."""

    def __init__(self, config, run_name=None) -> None:
        pass

    def log_config(self, config) -> None:
        pass

    def log_metrics(self, metrics, step=None) -> None:
        pass

    def log_train_step(self, agent_id, metrics, step) -> None:
        _SAMPLES.append((time.monotonic(), int(step), int(metrics.get("chunks_received", 0))))

    def finish(self) -> None:
        pass


def _make_config(num_workers: int, checkpoint_dir: str):
    from colosseum.core.config import ColosseumConfig, load_config

    data = load_config(CONFIG).model_dump()
    data["rollout"].update(
        num_workers=num_workers,
        envs_per_worker=ENVS_PER_WORKER,
        chunk_length=CHUNK_LENGTH,
        match_refresh_interval_sec=0.0,
    )
    data["learner"].update(batch_chunks=BATCH_CHUNKS, queue_size=4 * BATCH_CHUNKS, device="cpu")
    data["training"]["total_timesteps"] = 10**12  # stopped by the timer, not the budget
    data["self_play"]["checkpoint_interval"] = 10**9  # no checkpoints: every slot collects
    data["metrics"].update(use_wandb=False, log_interval=1)
    data["checkpoint"]["dir"] = checkpoint_dir
    return ColosseumConfig(**data)


def _rates(samples: list[tuple[float, int, int]], start: float, end: float, num_players: int) -> dict:
    window = [s for s in samples if start <= s[0] <= end]
    if len(window) < 2:
        return {"updates_per_s": 0.0, "env_steps_per_s": 0.0, "train_steps": 0}
    (t0, step0, chunks0), (t1, step1, chunks1) = window[0], window[-1]
    dt = max(t1 - t0, 1e-9)
    return {
        "updates_per_s": (step1 - step0) / dt,
        "env_steps_per_s": (chunks1 - chunks0) * CHUNK_LENGTH / num_players / dt,
        "train_steps": step1 - step0,
    }


def run_one(num_workers: int, duration: float, warmup: float) -> dict:
    import colosseum.launcher as launcher_mod

    _SAMPLES.clear()
    launcher_mod.WandBLogger = _Recorder
    with tempfile.TemporaryDirectory(prefix="bench-") as tmp:
        config = _make_config(num_workers, tmp)
        launcher = launcher_mod.Launcher(config)
        started = time.monotonic()
        timer = threading.Timer(warmup + duration, launcher._stop_event.set)
        timer.start()
        try:
            launcher.launch()
        finally:
            timer.cancel()
    result = _rates(_SAMPLES, started + warmup, started + warmup + duration, config.env.num_players)
    result["workers"] = num_workers
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--duration", type=float, default=60.0, help="measured seconds per run")
    parser.add_argument("--warmup", type=float, default=15.0, help="ignored seconds after start")
    parser.add_argument("--json", type=str, default=None, help="also write results to this JSON file")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.WARNING)

    import torch

    machine = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads_default": torch.get_num_threads(),
    }
    print(f"machine: {machine}", flush=True)
    print(
        f"config: {CONFIG.relative_to(REPO_ROOT)}, envs_per_worker={ENVS_PER_WORKER}, "
        f"chunk_length={CHUNK_LENGTH}, batch_chunks={BATCH_CHUNKS}, "
        f"warmup={args.warmup:.0f}s, duration={args.duration:.0f}s",
        flush=True,
    )

    results = []
    for n in args.workers:
        r = run_one(n, args.duration, args.warmup)
        results.append(r)
        print(
            f"workers={n}: updates/s={r['updates_per_s']:.2f} env_steps/s={r['env_steps_per_s']:.0f} "
            f"(train_steps in window={r['train_steps']})",
            flush=True,
        )

    print()
    print("| workers | updates/s | env steps/s |")
    print("|---|---|---|")
    for r in results:
        print(f"| {r['workers']} | {r['updates_per_s']:.2f} | {r['env_steps_per_s']:.0f} |")
    rates = [r["env_steps_per_s"] for r in results]
    monotonic = all(b > a for a, b in zip(rates, rates[1:]))
    print(f"\nenv steps/s increases monotonically with workers: {'yes' if monotonic else 'no'}")

    if args.json:
        Path(args.json).write_text(json.dumps({"machine": machine, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test it (about 1 minute)**

Run: `.venv/bin/python scripts/bench_throughput.py --workers 1 2 --duration 10 --warmup 8`

Expected: two `workers=N: updates/s=... env_steps/s=...` lines with non-zero rates, the markdown table, and the monotonic line. Worker processes may print tracebacks ending in `FileNotFoundError` from `torch/multiprocessing/reductions.py` during shutdown (R6-02, fixed in T2.2); they do not affect the measurement.

- [ ] **Step 3: Lint**

Run: `.venv/bin/ruff check scripts` — expected `All checks passed!`

- [ ] **Step 4: Measure the baseline (about 4 minutes; keep the machine otherwise idle)**

Run: `.venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15 --json /tmp/bench_before.json`

Expected: three result lines and `/tmp/bench_before.json`. Do not "fix" bad scaling here: it is the baseline.

- [ ] **Step 5: Write `docs/benchmarks.md` from the measurement**

Run (renders the Russian document from the JSON, so no number is typed by hand):

```bash
.venv/bin/python - <<'EOF'
import json
import subprocess
from datetime import date
from pathlib import Path

data = json.loads(Path("/tmp/bench_before.json").read_text())
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
m = data["machine"]
rows = "\n".join(
    f"| {r['workers']} | {r['updates_per_s']:.2f} | {r['env_steps_per_s']:.0f} |" for r in data["results"]
)
rates = [r["env_steps_per_s"] for r in data["results"]]
monotonic = all(b > a for a, b in zip(rates, rates[1:]))
note = "" if monotonic else (
    "Вероятная причина — переподписка потоков torch в каждом процессе (R2-04, R6-04); "
    "исправляется в T2.1.\n"
)
text = f"""# Замеры производительности

Throughput обучения на крестиках-ноликах (`configs/examples/tic_tac_toe.yaml`, self-play,
8 сред на воркер, `chunk_length=32`, `batch_chunks=8`, без чекпоинтов, каждое место
собирает данные). Скрипт: `scripts/bench_throughput.py`. Каждый прогон — настоящий
`Launcher` (spawn), 15 с прогрева не учитываются, затем 60 с замера.

- **апдейты/с** — шаги обучения лёрнера в секунду;
- **шаги сред/с** — шаги сред, потреблённые лёрнером в секунду
  (`chunks_received * chunk_length / num_players`); при полной очереди воркеры
  блокируются, поэтому потребление равно производству.

Команда (одна и та же для «до» и «после»):

```bash
.venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15
```

## До (SP1, блок 0, до исправления потоков torch)

- Дата: {date.today().isoformat()}, коммит `{commit}`.
- Машина: {m["platform"]}, {m["cpu_count"]} ядер, Python {m["python"]}, torch {m["torch"]},
  потоков torch по умолчанию: {m["torch_threads_default"]}.

| воркеры | апдейты/с | шаги сред/с |
|---|---|---|
{rows}

Рост шагов сред/с при 1 → 2 → 4 воркерах монотонный: {"да" if monotonic else "нет"}.
{note}
## После

Заполняется в T8.3 той же командой на той же машине.
"""
Path("docs/benchmarks.md").write_text(text)
print(text)
EOF
```

Expected: the printed document has three table rows matching Step 4.

- [ ] **Step 6: Run the fast suite**

Run: `.venv/bin/python -m pytest -m "not gpu and not slow" -q` — expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/bench_throughput.py docs/benchmarks.md
git commit -m "perf: add throughput benchmark and record the pre-SP1 baseline"
```

---

### Task T0.5: Extract `RolloutLoop` (pure refactor) + characterization tests

The plan moves this extraction from block 3 to block 0 (see "Deviation from the spec" in `00-overview.md`): every later contract test drives the real worker loop in-process. Behavior is unchanged, with one intentional exception: the initial weight sync at loop construction now also records the payload's `policy_version` (the old code loaded the weights but kept version 0 until the first periodic sync). Step 9 proves bit-identical chunks against the old implementation on a run without initial weights.

**Files:**
- Create: `src/colosseum/worker/rollout_loop.py` (`RolloutBuffer`, `LoopIO`, `RolloutLoop`, `LATEST_NETWORK_ID` and the helper functions, moved from `rollout_worker.py`)
- Modify: `src/colosseum/worker/rollout_worker.py` (rewrite: thin process wrapper)
- Modify: `src/colosseum/eval.py:180`, `src/colosseum/launcher.py:216` (imports from `rollout_loop`)
- Modify: `tests/unit/test_performance.py`, `tests/unit/test_recurrent.py`, `tests/unit/test_review_fixes.py` (imports from `rollout_loop`)
- Modify: `tests/helpers.py` (add `CountingEnv`)
- Create: `tests/contract/harness.py`, `tests/contract/test_rollout_loop_characterization.py`

**Interfaces:**
- Consumes: T0.2 (layout, `helpers.example_config`), T0.3 (ruff).
- Produces (contract `colosseum.worker.rollout_loop`, plus):
  - `LoopIO(send_chunk, poll_weights, report_result=None, poll_command=None, add_env_steps=None)` — `add_env_steps` is declared but not called until T2.5.
  - `RolloutLoop(*, worker_id, env_fn, num_envs, chunk_length, agent_ids, model_factories, io, gamma=0.99, weight_sync_interval=5.0, slot_agent_map=None, slot_network_map=None, collect_mask=None, checkpoint_state_dicts_by_agent=None, seed=None, vec_env_kind="sync", subproc_workers=None)`; `.step() -> int` (= `num_envs`), `.sync_weights()`, `.run(should_stop, max_env_steps=0)`, `.close()`, `.stats -> {"chunks_sent", "env_steps", "episodes", "parked_buffers"}` (`parked_buffers` is 0 until T2.6). `gamma` is stored for T3.3. In this task `model_factories` still return `ActorCriticNetwork`s.
  - Module-level names moved here: `LATEST_NETWORK_ID`, `RolloutBuffer`, `_run_inference_group`, `_apply_command`, `_build_chunk`, `_make_match_result` (was `_report_episode_result`, now returns the `MatchResult`), `_extract_action_masks`, `_extract_active_flags`.
  - `colosseum.worker.rollout_worker.rollout_worker_process(...)` — signature unchanged (still `network_factories`, renamed to `model_factories` in T1.5); adapts queues to `LoopIO`: chunks → `trajectory_queues[chunk.agent_id]` (blocking `put` that gives up once `stop_event` is set), weights and commands drained newest-wins with `_drain_latest(q)`, results `put_nowait` (dropped when full).
  - `tests/helpers.py`: `CountingEnv(num_players=2, episode_length=5, num_actions=3, obs_dim=4)` — deterministic simultaneous-move env; obs of player `p` at in-episode step `t` is `[t/episode_length, p, 1, 0, ...]`; reward 1 iff action == `(t + p) % num_actions`; terminates after `episode_length` steps.
  - `tests/contract/harness.py`: `OBS_DIM = 4`, `NUM_ACTIONS = 3`, `simple_factory()`, `counting_env_fn(num_players=2, episode_length=5)`, `weights_payload(agent_id, model, version) -> WeightPayload`, `LoopRecorder` (fields `chunks`, `results`, `pending_weights: dict[str, WeightPayload]`, `pending_commands: list[WorkerCommand]`; `.io() -> LoopIO`), `make_loop(*, agent_ids=None, model_factories=None, env_fn=None, num_envs=2, chunk_length=4, seed=123, initial_weights=None, **loop_kwargs) -> (RolloutLoop, LoopRecorder)`, `run_steps(loop, n)`, `run_until_chunks(loop, rec, n_chunks, max_steps=10_000) -> list[TrajectoryChunk]`, `step_index(chunk, episode_length=5) -> list[int]`, `player_index(chunk) -> set[int]`.

- [ ] **Step 1: Add `CountingEnv` to the test helpers**

Replace the whole of `tests/helpers.py` with:

```python
"""Shared test helpers: toy environments and small networks/models."""

from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist

REPO_ROOT = Path(__file__).resolve().parent.parent


def example_config(name: str) -> Path:
    """Absolute path of ``configs/examples/<name>`` (tests run with cwd = tmp_path)."""
    return REPO_ROOT / "configs" / "examples" / name


def make_simple_network(obs_dim=8, hidden_dim=16, num_actions=4):
    """Create a simple feedforward ActorCriticNetwork for testing."""
    from colosseum.networks.actor_critic import ActorCriticNetwork

    encoder = SimpleEncoder(obs_dim, hidden_dim)
    policy = SimplePolicy(hidden_dim, num_actions)
    value = SimpleValue(hidden_dim)
    return ActorCriticNetwork(encoder, policy, value)


class SimpleEncoder(BaseEncoder):
    def __init__(self, obs_dim=8, hidden_dim=16):
        super().__init__()
        self._latent_dim = hidden_dim
        self.fc = nn.Linear(obs_dim, hidden_dim)

    @property
    def latent_dim(self):
        return self._latent_dim

    def forward(self, obs):
        return torch.relu(self.fc(obs))


class SimplePolicy(BasePolicy):
    def __init__(self, hidden_dim=16, num_actions=4):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, latent):
        return CategoricalDist(self.fc(latent))


class SimpleValue(BaseValue):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, latent):
        return self.fc(latent).squeeze(-1)


# ---------------------------------------------------------------------------
# Deterministic toy environment for contract tests
# ---------------------------------------------------------------------------


class CountingEnv(BaseEnv):
    """Deterministic N-player simultaneous-move env for contract tests.

    Every episode lasts exactly ``episode_length`` steps and then terminates.
    At in-episode step ``t`` player ``p`` observes
    ``[t / episode_length, p, 1.0, 0.0, ...]`` (``obs_dim`` floats) and gets
    reward 1.0 if its action equals ``(t + p) % num_actions``, else 0.0.
    The step and player index can be recovered from any recorded observation:
    ``t = round(obs[0] * episode_length)``, ``p = round(obs[1])``.
    """

    def __init__(self, num_players: int = 2, episode_length: int = 5,
                 num_actions: int = 3, obs_dim: int = 4) -> None:
        if obs_dim < 3:
            raise ValueError("obs_dim must be >= 3")
        self._num_players = num_players
        self.episode_length = episode_length
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self._t = 0

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(self.num_actions)

    def _obs(self, p: int) -> np.ndarray:
        o = np.zeros(self.obs_dim, dtype=np.float32)
        o[0] = self._t / self.episode_length
        o[1] = float(p)
        o[2] = 1.0
        return o

    def reset(self, seed=None):
        self._t = 0
        players = range(self._num_players)
        return {p: self._obs(p) for p in players}, {p: {} for p in players}

    def step(self, actions):
        players = range(self._num_players)
        rewards = {p: 1.0 if int(actions[p]) == (self._t + p) % self.num_actions else 0.0 for p in players}
        self._t += 1
        done = self._t >= self.episode_length
        obs = {p: self._obs(p) for p in players}
        return obs, rewards, {p: done for p in players}, {p: False for p in players}, {p: {} for p in players}
```

- [ ] **Step 2: Write the harness**

Create `tests/contract/harness.py`:

```python
"""Helpers to drive the real ``RolloutLoop`` in-process (no worker processes)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from helpers import CountingEnv, make_simple_network

OBS_DIM = 4
NUM_ACTIONS = 3


def simple_factory() -> Any:
    """Model factory used by the contract tests (small MLP actor-critic)."""
    return make_simple_network(obs_dim=OBS_DIM, hidden_dim=16, num_actions=NUM_ACTIONS)


def counting_env_fn(num_players: int = 2, episode_length: int = 5) -> Callable[[], CountingEnv]:
    return partial(CountingEnv, num_players=num_players, episode_length=episode_length,
                   num_actions=NUM_ACTIONS, obs_dim=OBS_DIM)


def weights_payload(agent_id: str, model: Any, version: int) -> WeightPayload:
    """Weight payload carrying ``model``'s current parameters."""
    return WeightPayload(
        agent_id=agent_id,
        policy_version=version,
        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
    )


@dataclass
class LoopRecorder:
    """In-memory endpoints for ``LoopIO``: collects outputs, serves queued inputs."""

    chunks: list[TrajectoryChunk] = field(default_factory=list)
    results: list[MatchResult] = field(default_factory=list)
    pending_weights: dict[str, WeightPayload] = field(default_factory=dict)
    pending_commands: list[WorkerCommand] = field(default_factory=list)

    def poll_weights(self, agent_id: str) -> WeightPayload | None:
        return self.pending_weights.pop(agent_id, None)

    def poll_command(self) -> WorkerCommand | None:
        return self.pending_commands.pop(0) if self.pending_commands else None

    def io(self) -> LoopIO:
        return LoopIO(
            send_chunk=self.chunks.append,
            poll_weights=self.poll_weights,
            report_result=self.results.append,
            poll_command=self.poll_command,
        )


def make_loop(
    *,
    agent_ids: list[str] | None = None,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    env_fn: Callable[[], Any] | None = None,
    num_envs: int = 2,
    chunk_length: int = 4,
    seed: int = 123,
    initial_weights: dict[str, WeightPayload] | None = None,
    **loop_kwargs: Any,
) -> tuple[RolloutLoop, LoopRecorder]:
    """Build a ``RolloutLoop`` wired to a fresh ``LoopRecorder``."""
    agent_ids = agent_ids or ["agent_0"]
    if model_factories is None:
        model_factories = {aid: simple_factory for aid in agent_ids}
    rec = LoopRecorder()
    if initial_weights:
        rec.pending_weights.update(initial_weights)
    loop = RolloutLoop(
        worker_id=0,
        env_fn=env_fn or counting_env_fn(),
        num_envs=num_envs,
        chunk_length=chunk_length,
        agent_ids=agent_ids,
        model_factories=model_factories,
        io=rec.io(),
        seed=seed,
        **loop_kwargs,
    )
    return loop, rec


def run_steps(loop: RolloutLoop, n: int) -> None:
    for _ in range(n):
        loop.step()


def run_until_chunks(loop: RolloutLoop, rec: LoopRecorder, n_chunks: int,
                     max_steps: int = 10_000) -> list[TrajectoryChunk]:
    """Step until at least ``n_chunks`` chunks were sent; return the first ``n_chunks``."""
    for _ in range(max_steps):
        if len(rec.chunks) >= n_chunks:
            return rec.chunks[:n_chunks]
        loop.step()
    raise AssertionError(f"only {len(rec.chunks)} chunks after {max_steps} steps")


def step_index(chunk: TrajectoryChunk, episode_length: int = 5) -> list[int]:
    """In-episode step index of every transition (decoded from CountingEnv obs)."""
    return [round(float(x) * episode_length) for x in chunk.observations[:, 0]]


def player_index(chunk: TrajectoryChunk) -> set[int]:
    """Set of player indices whose observations appear in ``chunk``."""
    return {round(float(x)) for x in chunk.observations[:, 1]}
```

- [ ] **Step 3: Write the characterization tests**

Create `tests/contract/test_rollout_loop_characterization.py`:

```python
"""Characterization tests for RolloutLoop (driven in-process, no worker processes)."""

from __future__ import annotations

import numpy as np
import torch

from colosseum.core.types import WorkerCommand
from harness import (
    NUM_ACTIONS,
    make_loop,
    player_index,
    run_steps,
    simple_factory,
    step_index,
    weights_payload,
)

EPISODE = 5


def test_chunk_count_shapes_and_stats():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 20)  # 20 transitions per slot, 4 slots -> 5 chunks per slot
    loop.close()

    assert len(rec.chunks) == 20
    assert loop.stats["chunks_sent"] == 20
    assert loop.stats["env_steps"] == 40
    for c in rec.chunks:
        assert c.agent_id == "agent_0"
        assert c.observations.shape == (4, 4)
        assert c.actions.shape == (4,) and c.actions.dtype == torch.int64
        assert c.action_log_probs.shape == (4,)
        assert c.values.shape == (4,)
        assert c.rewards.shape == (4,)
        assert c.dones.shape == (4,)
        assert c.bootstrap_value.shape == ()
        assert torch.isfinite(c.action_log_probs).all() and (c.action_log_probs <= 0).all()
        assert torch.isfinite(c.values).all()
        assert c.behavior_policy_version == 0
        assert c.action_masks is None


def test_transitions_are_consecutive_and_rewards_dones_align():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 20)
    loop.close()

    for c in rec.chunks:
        ts = step_index(c, EPISODE)
        (p,) = player_index(c)  # every chunk belongs to exactly one slot
        for i in range(len(ts) - 1):
            assert ts[i + 1] == (ts[i] + 1) % EPISODE
        for i, t in enumerate(ts):
            assert bool(c.dones[i]) == (t == EPISODE - 1)
            expected_reward = 1.0 if int(c.actions[i]) == (t + p) % NUM_ACTIONS else 0.0
            assert float(c.rewards[i]) == expected_reward
        if bool(c.dones[-1]):
            assert float(c.bootstrap_value) == 0.0


def test_multi_agent_routing_by_slot():
    torch.manual_seed(0)
    loop, rec = make_loop(
        agent_ids=["a", "b"], num_envs=2, chunk_length=4,
        slot_agent_map=[["a", "b"], ["a", "b"]],
    )
    run_steps(loop, 8)
    loop.close()

    by_agent = {"a": 0, "b": 0}
    for c in rec.chunks:
        (p,) = player_index(c)
        assert c.agent_id == ("a" if p == 0 else "b")
        by_agent[c.agent_id] += 1
    assert by_agent == {"a": 4, "b": 4}


def test_non_collecting_checkpoint_slot_produces_no_chunks():
    torch.manual_seed(0)
    ckpt = {k: v.clone() for k, v in simple_factory().state_dict().items()}
    loop, rec = make_loop(
        num_envs=2, chunk_length=4,
        collect_mask=[[True, False], [True, False]],
        slot_network_map=[["latest", "ckpt_v1"], ["latest", "ckpt_v1"]],
        checkpoint_state_dicts_by_agent={"agent_0": {"ckpt_v1": ckpt}},
    )
    run_steps(loop, 8)
    loop.close()

    assert len(rec.chunks) == 4
    assert all(player_index(c) == {0} for c in rec.chunks)


def test_episode_results_are_reported_per_env():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 20)
    loop.close()

    assert len(rec.results) == 8  # 2 envs x 4 episodes of 5 steps
    for r in rec.results:
        assert r.episode_length == EPISODE
        assert set(r.player_outcomes) == {"agent_0:latest"}  # both seats share one key
        assert set(r.total_rewards) == {"agent_0:latest"}
    assert loop.stats["episodes"] == 8


def test_initial_and_periodic_weight_sync_set_policy_version():
    torch.manual_seed(0)
    src = simple_factory()
    loop, rec = make_loop(
        num_envs=1, chunk_length=4, weight_sync_interval=0.0,
        initial_weights={"agent_0": weights_payload("agent_0", src, 7)},
    )
    run_steps(loop, 4)
    assert [c.behavior_policy_version for c in rec.chunks] == [7, 7]

    rec.pending_weights["agent_0"] = weights_payload("agent_0", src, 9)
    run_steps(loop, 1)  # the sync at the end of this step picks up version 9
    run_steps(loop, 3)
    loop.close()
    assert rec.chunks[-1].behavior_policy_version == 9
    latest = loop._networks["agent_0"]["latest"]
    for k, v in src.state_dict().items():
        assert torch.equal(latest.state_dict()[k], v)


def test_command_reassignment_applies_at_episode_boundary():
    torch.manual_seed(0)
    loop, rec = make_loop(num_envs=2, chunk_length=4)
    run_steps(loop, 10)  # two full episodes in each env
    rec.pending_commands.append(WorkerCommand(
        slot_agent_map=[["agent_0", "agent_0"], ["agent_0", "agent_0"]],
        slot_network_map=[["latest", "latest"], ["latest", "latest"]],
        collect_mask=[[True, False], [True, False]],
        new_checkpoints={},
    ))
    sent_before_boundary = None
    for i in range(25):
        loop.step()
        if i == 4:  # the 3rd episode ends on this step; the command applies here
            sent_before_boundary = len(rec.chunks)
    loop.close()

    after = rec.chunks[sent_before_boundary:]
    assert after, "player 0 must keep producing chunks"
    assert all(player_index(c) == {0} for c in after)


def test_run_respects_max_env_steps_and_stop():
    torch.manual_seed(0)
    loop, _ = make_loop(num_envs=2, chunk_length=4)
    loop.run(should_stop=lambda: True)
    assert loop.stats["env_steps"] == 0
    loop.run(should_stop=lambda: False, max_env_steps=12)
    assert loop.stats["env_steps"] == 12
    loop.close()


def test_same_seed_gives_identical_chunks():
    def collect():
        torch.manual_seed(0)
        loop, rec = make_loop(num_envs=2, chunk_length=4, seed=7)
        run_steps(loop, 12)
        loop.close()
        return rec.chunks

    a, b = collect(), collect()
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert torch.equal(x.actions, y.actions)
        assert torch.equal(x.observations, y.observations)
        assert np.allclose(x.action_log_probs.numpy(), y.action_log_probs.numpy())
```

- [ ] **Step 4: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/contract -v`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.worker.rollout_loop'`.

- [ ] **Step 5: Save the old worker for the equivalence check**

Run: `git show HEAD:src/colosseum/worker/rollout_worker.py > /tmp/old_rollout_worker.py`

- [ ] **Step 6: Create `src/colosseum/worker/rollout_loop.py`**

The class body is the old `rollout_worker_process` loop split into `__init__` (everything before `while not stop_event.is_set()`) and `step()` (one loop iteration); local variables became attributes, queue calls became `LoopIO` callbacks.

```python
"""In-process rollout loop: vectorized envs, batched inference, trajectory chunks.

``RolloutLoop`` owns the environments and the per-agent network pools of one
worker. All I/O (sending chunks, pulling weights, reporting match results,
receiving match re-assignments) goes through the ``LoopIO`` callbacks, so the
loop can be driven step by step inside a test process. ``rollout_worker_process``
(``colosseum.worker.rollout_worker``) wraps it with multiprocessing queues.

Multi-agent: several agents can occupy different player slots of the same
environments. Inference is grouped by (agent_id, network_id) for batching, and
each chunk is routed to the agent that produced it.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from colosseum.core.action_spec import ActionSpec
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv

logger = logging.getLogger(__name__)

LATEST_NETWORK_ID = "latest"


@dataclass
class RolloutBuffer:
    """Pre-allocated trajectory buffer for a single (env, player) pair.

    Uses numpy arrays allocated once at creation with a write cursor.
    Eliminates per-step list append overhead and reduces copies at chunk build time.
    """

    chunk_length: int
    obs_shape: tuple = ()
    action_shape: tuple = ()
    action_dtype: type = np.int64
    num_actions: int = 0

    _observations: np.ndarray = field(init=False, repr=False)
    _actions: np.ndarray = field(init=False, repr=False)
    _log_probs: np.ndarray = field(init=False, repr=False)
    _rewards: np.ndarray = field(init=False, repr=False)
    _dones: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _action_masks: np.ndarray | None = field(init=False, default=None, repr=False)
    _cursor: int = field(init=False, default=0)
    _has_masks: bool = field(init=False, default=False)
    _lstm_h_init: torch.Tensor | None = field(init=False, default=None, repr=False)
    _lstm_c_init: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        T = self.chunk_length
        self._observations = np.zeros((T, *self.obs_shape), dtype=np.float32)
        self._actions = np.zeros((T, *self.action_shape), dtype=self.action_dtype)
        self._log_probs = np.zeros(T, dtype=np.float32)
        self._rewards = np.zeros(T, dtype=np.float32)
        self._dones = np.zeros(T, dtype=np.float32)
        self._values = np.zeros(T, dtype=np.float32)
        if self.num_actions > 0:
            self._action_masks = np.zeros((T, self.num_actions), dtype=np.bool_)

    def append(self, obs, action, log_prob, reward, done, value, action_mask=None):
        i = self._cursor
        self._observations[i] = obs
        self._actions[i] = action
        self._log_probs[i] = log_prob
        self._rewards[i] = reward
        self._dones[i] = done
        self._values[i] = value
        if action_mask is not None and self._action_masks is not None:
            self._action_masks[i] = action_mask
            self._has_masks = True
        self._cursor += 1

    @property
    def is_full(self) -> bool:
        return self._cursor >= self.chunk_length

    @property
    def steps(self) -> int:
        return self._cursor

    @property
    def has_masks(self) -> bool:
        return self._has_masks

    def set_lstm_init(self, h: torch.Tensor, c: torch.Tensor):
        """Save initial LSTM hidden state for the current chunk."""
        self._lstm_h_init = h.clone()
        self._lstm_c_init = c.clone()

    def reset(self):
        self._cursor = 0
        self._has_masks = False
        self._lstm_h_init = None
        self._lstm_c_init = None


@dataclass
class LoopIO:
    """Callbacks through which a ``RolloutLoop`` talks to the outside world."""

    send_chunk: Callable[[TrajectoryChunk], None]
    poll_weights: Callable[[str], WeightPayload | None]
    report_result: Callable[[MatchResult], None] | None = None
    poll_command: Callable[[], WorkerCommand | None] | None = None
    add_env_steps: Callable[[int], None] | None = None  # global env-step budget (wired in T2.5)


def _run_inference_group(
    net,
    indices: list[tuple],
    obs_flat: np.ndarray,
    all_masks: np.ndarray | None,
    hidden_states: dict,
    out_actions: np.ndarray,
    out_log_probs: np.ndarray,
    out_values: np.ndarray,
) -> None:
    """Batched inference for one (agent, network) group; writes into output arrays."""
    idx_list = [i[0] for i in indices]
    obs_batch = torch.from_numpy(np.ascontiguousarray(obs_flat[idx_list])).float()
    mask_batch = None
    if all_masks is not None:
        mask_batch = torch.from_numpy(np.ascontiguousarray(all_masks[idx_list])).bool()

    hidden_batch = None
    if net.is_recurrent:
        h_list, c_list = [], []
        for _, env_idx, p in indices:
            h, c = hidden_states.get((env_idx, p), net.initial_hidden(1))
            h_list.append(h)
            c_list.append(c)
        hidden_batch = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))

    with torch.no_grad():
        actions, log_probs, values, new_hidden = net.act(
            obs_batch, action_mask=mask_batch, hidden=hidden_batch,
        )

    if net.is_recurrent and new_hidden is not None:
        for j, (_, env_idx, p) in enumerate(indices):
            hidden_states[(env_idx, p)] = (
                new_hidden[0][:, j:j + 1, :].clone(),
                new_hidden[1][:, j:j + 1, :].clone(),
            )

    idx_arr = np.asarray(idx_list, dtype=np.intp)
    out_actions[idx_arr] = actions.numpy()
    out_log_probs[idx_arr] = log_probs.numpy().astype(np.float32, copy=False)
    out_values[idx_arr] = values.numpy().astype(np.float32, copy=False)


def _apply_command(cmd, networks_by_agent, network_factories, pending) -> None:
    """Load any new checkpoints into the pool and stash the new slot maps.

    The slot maps are applied per-env at the next episode boundary (so a match
    keeps a consistent assignment for its whole episode).
    """
    for aid, ckpts in cmd.new_checkpoints.items():
        if aid not in networks_by_agent:
            continue
        for ckpt_id, sd in ckpts.items():
            if ckpt_id not in networks_by_agent[aid]:
                net = network_factories[aid]()
                net.load_state_dict(sd)
                net.eval()
                networks_by_agent[aid][ckpt_id] = net
    if cmd.slot_agent_map:
        pending["slot_agent_map"] = cmd.slot_agent_map
        pending["slot_network_map"] = cmd.slot_network_map
        pending["collect_mask"] = cmd.collect_mask


def _build_chunk(buffer: RolloutBuffer, agent_id, bootstrap_value, policy_version):
    """Build a TrajectoryChunk from a pre-allocated buffer."""
    chunk = TrajectoryChunk(
        agent_id=agent_id,
        observations=torch.from_numpy(buffer._observations.copy()),
        actions=torch.from_numpy(buffer._actions.copy()),
        action_log_probs=torch.from_numpy(buffer._log_probs.copy()),
        rewards=torch.from_numpy(buffer._rewards.copy()),
        dones=torch.from_numpy(buffer._dones.copy()),
        values=torch.from_numpy(buffer._values.copy()),
        bootstrap_value=torch.tensor(bootstrap_value, dtype=torch.float32),
        behavior_policy_version=policy_version,
    )
    if buffer._has_masks and buffer._action_masks is not None:
        chunk.action_masks = torch.from_numpy(buffer._action_masks.copy())
    if buffer._lstm_h_init is not None:
        chunk.lstm_hidden = (buffer._lstm_h_init, buffer._lstm_c_init)
    return chunk


def _make_match_result(
    worker_id: int,
    env_idx: int,
    step: int,
    ep_rewards: np.ndarray,
    ep_length: int,
    slot_nets: list[str],
    slot_agent_ids: list[str],
    terminal_infos: dict[int, dict] | None = None,
) -> MatchResult:
    """Build the episode result reported to the coordinator.

    Player outcomes are keyed by ``agent_id:network_id`` (``"agent_0:latest"``,
    ``"agent_0:ckpt_v100"``). Outcomes prefer the env's authoritative signal
    (``rank``/``outcome`` in the terminal info) and fall back to cumulative reward.
    """
    from colosseum.core.outcomes import player_outcomes as _player_outcomes

    num_players = len(slot_nets)
    outcomes = _player_outcomes(ep_rewards, terminal_infos, num_players)

    player_outcomes: dict[str, float] = {}
    total_rewards: dict[str, float] = {}
    for p in range(num_players):
        player_key = f"{slot_agent_ids[p]}:{slot_nets[p]}"
        total_rewards[player_key] = float(ep_rewards[p])
        player_outcomes[player_key] = float(outcomes[p])

    return MatchResult(
        match_id=f"w{worker_id}_e{env_idx}_{step}",
        player_outcomes=player_outcomes,
        total_rewards=total_rewards,
        episode_length=int(ep_length),
    )


def _extract_action_masks(
    infos: list[dict],
    num_envs: int,
    num_players: int,
    action_spec=None,
) -> np.ndarray | None:
    """Extract action masks from env info dicts into a flat array.

    Convention: info[env_idx][player_idx]["action_mask"] is a bool ndarray
    or dict of per-component bool ndarrays (for composite action spaces).
    Returns [num_envs * num_players, num_actions] bool array, or None if no masks.
    """
    if not infos:
        return None
    first_info = infos[0]
    if not isinstance(first_info, dict) or 0 not in first_info:
        return None
    if "action_mask" not in first_info[0]:
        return None

    masks = []
    for env_idx in range(num_envs):
        for p in range(num_players):
            raw = infos[env_idx][p]["action_mask"]
            if isinstance(raw, dict) and action_spec is not None:
                masks.append(action_spec.flatten_mask(raw))
            else:
                masks.append(raw)
    return np.array(masks, dtype=bool)


def _extract_active_flags(
    infos: list[dict],
    num_envs: int,
    num_players: int,
) -> np.ndarray | None:
    """Per-slot ``info["active"]`` flags, or None if the env doesn't provide them."""
    if not infos:
        return None
    first = infos[0]
    if not isinstance(first, dict) or 0 not in first:
        return None
    if not isinstance(first[0], dict) or "active" not in first[0]:
        return None
    flags = np.ones((num_envs, num_players), dtype=bool)
    for e in range(num_envs):
        for p in range(num_players):
            flags[e][p] = bool(infos[e][p].get("active", True))
    return flags


class RolloutLoop:
    """One worker's environments + network pools, advanced one vector step at a time."""

    def __init__(
        self,
        *,
        worker_id: int,
        env_fn: Callable[[], BaseEnv],
        num_envs: int,
        chunk_length: int,
        agent_ids: list[str],
        model_factories: dict[str, Callable[[], nn.Module]],
        io: LoopIO,
        gamma: float = 0.99,
        weight_sync_interval: float = 5.0,
        slot_agent_map: list[list[str]] | None = None,
        slot_network_map: list[list[str]] | None = None,
        collect_mask: list[list[bool]] | None = None,
        checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
        seed: int | None = None,
        vec_env_kind: str = "sync",
        subproc_workers: int | None = None,
    ) -> None:
        if checkpoint_state_dicts_by_agent is None:
            checkpoint_state_dicts_by_agent = {aid: {} for aid in agent_ids}

        self.worker_id = worker_id
        self.num_envs = num_envs
        self.chunk_length = chunk_length
        self.agent_ids = list(agent_ids)
        self.gamma = gamma  # used for truncation bootstrapping (T3.3)
        self._io = io
        self._model_factories = model_factories
        self._weight_sync_interval = weight_sync_interval

        logger.info(
            f"Worker {worker_id}: starting with {num_envs} envs ({vec_env_kind}), "
            f"chunk_length={chunk_length}, agents={agent_ids}"
        )

        if vec_env_kind == "subprocess":
            from colosseum.envs.subproc_vec_env import SubprocessVectorEnv
            self._vec_env = SubprocessVectorEnv(env_fn, num_envs, num_workers=subproc_workers)
        else:
            self._vec_env = VectorEnv(env_fn, num_envs)
        num_players = self._vec_env.num_players
        self.num_players = num_players

        # Per-agent network pools: "latest" receives weight updates; checkpoint
        # networks are frozen opponents.
        self._networks: dict[str, dict[str, nn.Module]] = {}
        self._policy_versions: dict[str, int] = {}
        for aid in self.agent_ids:
            nets: dict[str, nn.Module] = {}
            nets[LATEST_NETWORK_ID] = model_factories[aid]()
            nets[LATEST_NETWORK_ID].eval()
            self._policy_versions[aid] = 0
            ckpt_dicts = checkpoint_state_dicts_by_agent.get(aid, {})
            for ckpt_id, sd in ckpt_dicts.items():
                net = model_factories[aid]()
                net.load_state_dict(sd)
                net.eval()
                nets[ckpt_id] = net
            if ckpt_dicts:
                logger.info(
                    f"Worker {worker_id}: agent {aid}: loaded {len(ckpt_dicts)} "
                    f"checkpoint(s): {list(ckpt_dicts.keys())}"
                )
            self._networks[aid] = nets

        # Initial weights for every agent's latest network.
        self.sync_weights()

        self._any_recurrent = self._compute_any_recurrent()

        # Default slot assignment: every slot is the first agent's latest network.
        if slot_agent_map is None:
            slot_agent_map = [[self.agent_ids[0]] * num_players for _ in range(num_envs)]
        if slot_network_map is None:
            if collect_mask is not None:
                slot_network_map = []
                for e in range(num_envs):
                    slot_nets = []
                    for p in range(num_players):
                        aid = slot_agent_map[e][p]
                        ckpts = checkpoint_state_dicts_by_agent.get(aid, {})
                        if collect_mask[e][p] or not ckpts:
                            slot_nets.append(LATEST_NETWORK_ID)
                        else:
                            slot_nets.append(next(iter(ckpts)))
                    slot_network_map.append(slot_nets)
            else:
                slot_network_map = [[LATEST_NETWORK_ID] * num_players for _ in range(num_envs)]
        if collect_mask is None:
            collect_mask = [[True] * num_players for _ in range(num_envs)]

        # Live (mutable) match assignment; WorkerCommand updates are staged in
        # `_pending` and applied per env at its next episode boundary.
        self._slot_agent_map = [list(row) for row in slot_agent_map]
        self._slot_network_map = [list(row) for row in slot_network_map]
        self._collect_mask = [list(row) for row in collect_mask]
        self._pending: dict[str, list | None] = {
            "slot_agent_map": None, "slot_network_map": None, "collect_mask": None,
        }

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self._obs, self._infos = self._vec_env.reset_all(seed=seed)

        self._hidden_states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        if self._any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    aid = self._slot_agent_map[env_idx][p]
                    net = self._networks[aid][LATEST_NETWORK_ID]
                    if net.is_recurrent:
                        self._hidden_states[(env_idx, p)] = net.initial_hidden(1)

        self._action_spec = ActionSpec.from_space(self._vec_env.action_space)
        obs_shape = self._obs.shape[2:]
        # A buffer exists for every slot (not only collecting ones): runtime
        # re-assignment can flip a slot's collect flag.
        self._buffers: list[list[RolloutBuffer]] = [
            [
                RolloutBuffer(
                    chunk_length=chunk_length,
                    obs_shape=obs_shape,
                    action_shape=self._action_spec.action_shape,
                    action_dtype=self._action_spec.numpy_dtype,
                    num_actions=self._action_spec.flat_mask_size,
                )
                for _ in range(num_players)
            ]
            for _ in range(num_envs)
        ]

        self._total_steps = 0
        self._chunks_sent = 0
        self._episodes = 0
        self._last_weight_sync = time.time()
        self._ep_rewards = np.zeros((num_envs, num_players), dtype=np.float64)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int64)

        num_slots = num_envs * num_players
        self._all_actions = np.zeros(
            (num_slots, *self._action_spec.action_shape), dtype=self._action_spec.numpy_dtype,
        )
        self._all_log_probs = np.zeros(num_slots, dtype=np.float32)
        self._all_values = np.zeros(num_slots, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "chunks_sent": self._chunks_sent,
            "env_steps": self._total_steps,
            "episodes": self._episodes,
            "parked_buffers": 0,
        }

    def sync_weights(self) -> None:
        """Load the newest available weights into every agent's latest network."""
        for aid in self.agent_ids:
            payload = self._io.poll_weights(aid)
            if payload is not None:
                self._networks[aid][LATEST_NETWORK_ID].load_state_dict(payload.state_dict)
                self._policy_versions[aid] = payload.policy_version

    def run(self, should_stop: Callable[[], bool], max_env_steps: int = 0) -> None:
        """Step until ``should_stop()`` or until ``max_env_steps`` env steps (0 = no limit)."""
        while not should_stop():
            if max_env_steps > 0 and self._total_steps >= max_env_steps:
                break
            self.step()

    def close(self) -> None:
        self._vec_env.close()

    def step(self) -> int:
        """Advance every env by one step. Returns the number of env steps taken."""
        num_envs, num_players = self.num_envs, self.num_players
        obs = self._obs
        infos = self._infos

        cmd = self._io.poll_command() if self._io.poll_command is not None else None
        if cmd is not None:
            _apply_command(cmd, self._networks, self._model_factories, self._pending)
            self._any_recurrent = self._compute_any_recurrent()

        # Save the recurrent state at chunk start (before inference).
        if self._any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    buf = self._buffers[env_idx][p]
                    if (self._collect_mask[env_idx][p] and buf.steps == 0
                            and (env_idx, p) in self._hidden_states):
                        h, c = self._hidden_states[(env_idx, p)]
                        buf.set_lstm_init(h, c)

        # Group slots by (agent_id, network_id) for batched inference.
        net_groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
        for env_idx in range(num_envs):
            for p in range(num_players):
                flat_idx = env_idx * num_players + p
                aid = self._slot_agent_map[env_idx][p]
                net_id = self._slot_network_map[env_idx][p]
                net_groups[(aid, net_id)].append((flat_idx, env_idx, p))

        obs_flat = obs.reshape(-1, *obs.shape[2:])
        self._all_actions.fill(0)
        self._all_log_probs.fill(0)
        self._all_values.fill(0)

        # Masks and turn flags describe the state the agents are about to act on.
        all_masks = _extract_action_masks(infos, num_envs, num_players, action_spec=self._action_spec)
        active_flags = _extract_active_flags(infos, num_envs, num_players)

        for (aid, net_id), indices in net_groups.items():
            agent_nets = self._networks[aid]
            net = agent_nets.get(net_id, agent_nets[LATEST_NETWORK_ID])
            _run_inference_group(
                net, indices, obs_flat, all_masks, self._hidden_states,
                self._all_actions, self._all_log_probs, self._all_values,
            )

        actions_np = self._all_actions.reshape(num_envs, num_players, *self._action_spec.action_shape)
        log_probs_np = self._all_log_probs.reshape(num_envs, num_players)
        values_np = self._all_values.reshape(num_envs, num_players)

        next_obs, rewards, terminated, truncated, infos = self._vec_env.step(actions_np)

        for env_idx in range(num_envs):
            done = terminated[env_idx] or truncated[env_idx]
            self._ep_lengths[env_idx] += 1

            for player_idx in range(num_players):
                self._ep_rewards[env_idx, player_idx] += rewards[env_idx, player_idx]

                slot_active = active_flags is None or active_flags[env_idx][player_idx]
                if self._collect_mask[env_idx][player_idx] and slot_active:
                    buf = self._buffers[env_idx][player_idx]
                    slot_aid = self._slot_agent_map[env_idx][player_idx]

                    action = actions_np[env_idx, player_idx]
                    if isinstance(action, np.ndarray) and action.ndim == 0:
                        action = action.item()

                    mask_for_step = None
                    if all_masks is not None:
                        mask_for_step = all_masks[env_idx * num_players + player_idx]

                    buf.append(
                        obs=obs[env_idx, player_idx],
                        action=action,
                        log_prob=log_probs_np[env_idx, player_idx],
                        reward=rewards[env_idx, player_idx],
                        done=done,
                        value=values_np[env_idx, player_idx],
                        action_mask=mask_for_step,
                    )

                    if buf.is_full:
                        bootstrap_net = self._networks[slot_aid][LATEST_NETWORK_ID]
                        next_obs_t = torch.tensor(
                            next_obs[env_idx, player_idx], dtype=torch.float32,
                        ).unsqueeze(0)
                        bootstrap_hidden = self._hidden_states.get((env_idx, player_idx))
                        with torch.no_grad():
                            _, _, bootstrap_val, _ = bootstrap_net.act(
                                next_obs_t, hidden=bootstrap_hidden,
                            )
                        bootstrap_val = 0.0 if done else bootstrap_val.item()

                        chunk = _build_chunk(
                            buf, slot_aid, bootstrap_val, self._policy_versions[slot_aid],
                        )
                        self._io.send_chunk(chunk)
                        self._chunks_sent += 1
                        buf.reset()

            if done and self._io.report_result is not None:
                term_infos = {
                    p: infos[env_idx][p].get("terminal_info", {})
                    for p in range(num_players)
                }
                self._io.report_result(_make_match_result(
                    self.worker_id, env_idx, self._total_steps,
                    self._ep_rewards[env_idx], self._ep_lengths[env_idx],
                    self._slot_network_map[env_idx], self._slot_agent_map[env_idx],
                    term_infos,
                ))

            if done:
                self._episodes += 1
                self._ep_rewards[env_idx] = 0.0
                self._ep_lengths[env_idx] = 0
                self._apply_pending_assignment(env_idx)

                if self._any_recurrent:
                    for p in range(num_players):
                        if (env_idx, p) in self._hidden_states:
                            aid = self._slot_agent_map[env_idx][p]
                            net = self._networks[aid][LATEST_NETWORK_ID]
                            self._hidden_states[(env_idx, p)] = net.initial_hidden(1)

        self._obs = next_obs
        self._infos = infos
        self._total_steps += num_envs

        now = time.time()
        if now - self._last_weight_sync >= self._weight_sync_interval:
            self.sync_weights()
            self._last_weight_sync = now

        return num_envs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_any_recurrent(self) -> bool:
        return any(
            nets[LATEST_NETWORK_ID].is_recurrent for nets in self._networks.values()
        )

    def _apply_pending_assignment(self, env_idx: int) -> None:
        """Apply a staged match re-assignment to one env at its episode boundary."""
        pending = self._pending
        if pending["slot_agent_map"] is None or env_idx >= len(pending["slot_agent_map"]):
            return
        new_agents = pending["slot_agent_map"][env_idx]
        new_nets = pending["slot_network_map"][env_idx]
        new_collect = pending["collect_mask"][env_idx]
        for p in range(self.num_players):
            # Discard the partial buffer when a slot's agent or collect flag changes.
            if (self._slot_agent_map[env_idx][p] != new_agents[p]
                    or self._collect_mask[env_idx][p] != new_collect[p]):
                self._buffers[env_idx][p].reset()
            self._slot_agent_map[env_idx][p] = new_agents[p]
            self._slot_network_map[env_idx][p] = new_nets[p]
            self._collect_mask[env_idx][p] = new_collect[p]
```

- [ ] **Step 7: Rewrite `src/colosseum/worker/rollout_worker.py` as a thin wrapper**

Replace the whole file with:

```python
"""Rollout worker process: wires multiprocessing queues to a ``RolloutLoop``.

The loop itself (envs, inference, chunking) lives in
``colosseum.worker.rollout_loop`` and is testable in-process; this module only
adapts queues to the loop's ``LoopIO`` callbacks.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from collections.abc import Callable
from typing import Any

from colosseum.core.types import MatchResult, TrajectoryChunk
from colosseum.envs.base_env import BaseEnv
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop

logger = logging.getLogger(__name__)


def _drain_latest(q: Any) -> Any | None:
    """Return the newest item available on ``q`` (dropping older ones), or None."""
    if q is None:
        return None
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest


def rollout_worker_process(
    worker_id: int,
    env_fn: Callable[[], BaseEnv],
    num_envs: int,
    chunk_length: int,
    agent_ids: list[str],
    network_factories: dict[str, Callable[[], Any]],
    trajectory_queues: dict[str, mp.Queue],
    weight_queues: dict[str, mp.Queue],
    stop_event: mp.Event,
    weight_sync_interval: float = 5.0,
    total_timesteps: int = 0,
    checkpoint_state_dicts_by_agent: dict[str, dict[str, dict]] | None = None,
    slot_network_map: list[list[str]] | None = None,
    collect_mask: list[list[bool]] | None = None,
    slot_agent_map: list[list[str]] | None = None,
    results_queue: mp.Queue | None = None,
    seed: int | None = None,
    command_queue: mp.Queue | None = None,
    vec_env_kind: str = "sync",
    subproc_workers: int | None = None,
) -> None:
    """Worker process entry: build a ``RolloutLoop`` and run it until stopped.

    Args mirror ``RolloutLoop``; queues are adapted to ``LoopIO`` callbacks:
    chunks go to ``trajectory_queues[chunk.agent_id]`` (blocking put that gives
    up when ``stop_event`` is set), weights are drained newest-wins from
    ``weight_queues[agent_id]``, results are put non-blocking on
    ``results_queue`` and commands are drained newest-wins from ``command_queue``.
    ``total_timesteps`` limits this worker's env steps (0 = until stopped).
    """

    def send_chunk(chunk: TrajectoryChunk) -> None:
        tq = trajectory_queues[chunk.agent_id]
        while not stop_event.is_set():
            try:
                tq.put(chunk, timeout=1.0)
                return
            except queue.Full:
                continue

    def report_result(result: MatchResult) -> None:
        try:
            results_queue.put_nowait(result)
        except queue.Full:
            pass  # non-critical, never block the rollout

    io = LoopIO(
        send_chunk=send_chunk,
        poll_weights=lambda aid: _drain_latest(weight_queues[aid]),
        report_result=report_result if results_queue is not None else None,
        poll_command=(lambda: _drain_latest(command_queue)) if command_queue is not None else None,
    )

    loop = RolloutLoop(
        worker_id=worker_id,
        env_fn=env_fn,
        num_envs=num_envs,
        chunk_length=chunk_length,
        agent_ids=agent_ids,
        model_factories=network_factories,
        io=io,
        weight_sync_interval=weight_sync_interval,
        slot_agent_map=slot_agent_map,
        slot_network_map=slot_network_map,
        collect_mask=collect_mask,
        checkpoint_state_dicts_by_agent=checkpoint_state_dicts_by_agent,
        seed=seed,
        vec_env_kind=vec_env_kind,
        subproc_workers=subproc_workers,
    )
    try:
        loop.run(should_stop=stop_event.is_set, max_env_steps=total_timesteps)
    finally:
        loop.close()
        # Detach feeder threads for queues this worker produced to, so undrained
        # data (e.g. chunks a stopped learner never consumed) can't block exit.
        for tq in trajectory_queues.values():
            try:
                tq.cancel_join_thread()
            except AttributeError:
                pass
        if results_queue is not None:
            results_queue.cancel_join_thread()
    stats = loop.stats
    logger.info(
        f"Worker {worker_id}: finished. Steps={stats['env_steps']}, chunks_sent={stats['chunks_sent']}"
    )
```

- [ ] **Step 8: Point the remaining importers at the new module**

```bash
sed -i 's/from colosseum.worker.rollout_worker import _extract_action_masks/from colosseum.worker.rollout_loop import _extract_action_masks/' src/colosseum/eval.py
sed -i 's/from colosseum.worker.rollout_worker import LATEST_NETWORK_ID/from colosseum.worker.rollout_loop import LATEST_NETWORK_ID/' src/colosseum/launcher.py
sed -i 's/from colosseum.worker.rollout_worker import /from colosseum.worker.rollout_loop import /' \
  tests/unit/test_performance.py tests/unit/test_recurrent.py tests/unit/test_review_fixes.py
```

Check: `grep -rn "rollout_worker import" src tests` only shows `rollout_worker_process` imports in `src/colosseum/launcher.py` and `src/colosseum/distributed.py`.

- [ ] **Step 9: Prove the refactor is pure**

Save as `/tmp/equiv_t05.py` and run it with `.venv/bin/python /tmp/equiv_t05.py`:

```python
"""Pure-refactor check: the old worker function and RolloutLoop produce identical chunks."""
import importlib.util
import queue
import sys
import threading
from functools import partial

import torch

sys.path.insert(0, "tests")
from helpers import CountingEnv, make_simple_network  # noqa: E402

from colosseum.worker.rollout_worker import rollout_worker_process as new_worker  # noqa: E402

spec = importlib.util.spec_from_file_location("old_rollout_worker", "/tmp/old_rollout_worker.py")
old = importlib.util.module_from_spec(spec)
sys.modules["old_rollout_worker"] = old
spec.loader.exec_module(old)


class ListQ:
    def __init__(self):
        self.items = []

    def put(self, x, timeout=None):
        self.items.append(x)

    def put_nowait(self, x):
        self.items.append(x)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def cancel_join_thread(self):
        pass


def run(worker_fn):
    torch.manual_seed(0)
    tq, wq, rq = {"agent_0": ListQ()}, {"agent_0": ListQ()}, ListQ()
    env_fn = partial(CountingEnv, num_players=2, episode_length=5, num_actions=3, obs_dim=4)
    worker_fn(0, env_fn, 2, 4, ["agent_0"], {"agent_0": lambda: make_simple_network(4, 16, 3)},
              tq, wq, threading.Event(), total_timesteps=40, seed=123, results_queue=rq)
    return tq["agent_0"].items, rq.items


a, ra = run(old.rollout_worker_process)
b, rb = run(new_worker)
assert len(a) == len(b) == 20, (len(a), len(b))
for x, y in zip(a, b):
    for f in ("observations", "actions", "action_log_probs", "rewards", "dones", "values", "bootstrap_value"):
        assert torch.equal(getattr(x, f), getattr(y, f)), f
assert [r.player_outcomes for r in ra] == [r.player_outcomes for r in rb]
print(f"identical: {len(a)} chunks, {len(ra)} results")
```

Expected: `identical: 20 chunks, 8 results`. Then `rm /tmp/old_rollout_worker.py /tmp/equiv_t05.py`.

- [ ] **Step 10: Run the characterization tests**

Run: `.venv/bin/python -m pytest tests/contract -v`

Expected: 9 passed.

- [ ] **Step 11: Run the process-level tests that use the worker wrapper**

Run: `.venv/bin/python -m pytest tests/integration/test_pipelines.py -v -m "not gpu"` (includes the slow pipelines, about 1 minute)

Expected: all pass.

- [ ] **Step 12: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!`; all tests pass.

- [ ] **Step 13: Commit**

```bash
git add src/colosseum/worker src/colosseum/eval.py src/colosseum/launcher.py tests
git commit -m "refactor: extract in-process RolloutLoop from the worker process with characterization tests"
```

---

### Task T1.1: `State` pytree utilities

**Files:**
- Create: `src/colosseum/networks/state.py`
- Test: `tests/unit/test_state.py`

**Interfaces:**
- Consumes: nothing (torch, numpy).
- Produces (contract `colosseum.networks.state`): `State`, `tree_map(fn, state)`, `tree_leaves(state) -> list[Tensor]`, `batch_size_of(state) -> int | None`, `slice_batch(state, idx)` (an `int` index keeps the batch dim; also accepts a sequence of ints, a long tensor or a bool mask), `cat_batch(states)` (all-`None` → `None`; mismatched structures → `ValueError`), `where_done(done, reset, state)`, `state_to_numpy(state)` (detached CPU copies), `state_from_numpy(obj, device="cpu")`, `state_to(state, device)`. Namedtuples keep their type through every function. Unknown node types raise `TypeError`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_state.py`:

```python
"""Unit tests for colosseum.networks.state (State pytree utilities)."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
import torch

from colosseum.networks.state import (
    batch_size_of,
    cat_batch,
    slice_batch,
    state_from_numpy,
    state_to,
    state_to_numpy,
    tree_leaves,
    tree_map,
    where_done,
)

Pair = namedtuple("Pair", ["a", "b"])


def _nested(batch: int = 3) -> dict:
    return {
        "h": torch.arange(batch * 2 * 4, dtype=torch.float32).reshape(batch, 2, 4),
        "pair": Pair(torch.ones(batch, 1), [torch.zeros(batch, dtype=torch.long)]),
        "none": None,
    }


def _assert_tree_equal(x, y):
    lx, ly = tree_leaves(x), tree_leaves(y)
    assert len(lx) == len(ly)
    for a, b in zip(lx, ly):
        assert a.dtype == b.dtype
        assert torch.equal(a, b)


def test_tree_map_keeps_structure_and_namedtuples():
    s = _nested()
    out = tree_map(lambda t: t * 2, s)
    assert set(out) == {"h", "pair", "none"}
    assert isinstance(out["pair"], Pair)
    assert isinstance(out["pair"].b, list)
    assert out["none"] is None
    assert torch.equal(out["h"], s["h"] * 2)
    assert tree_map(lambda t: t, None) is None


def test_tree_map_rejects_unknown_nodes():
    with pytest.raises(TypeError):
        tree_map(lambda t: t, {"x": 1.0})


def test_tree_leaves_order_is_insertion_order():
    s = _nested()
    leaves = tree_leaves(s)
    assert [tuple(t.shape) for t in leaves] == [(3, 2, 4), (3, 1), (3,)]
    assert tree_leaves(None) == []


def test_batch_size_of():
    assert batch_size_of(_nested(5)) == 5
    assert batch_size_of(None) is None
    assert batch_size_of({"x": None}) is None
    with pytest.raises(ValueError, match="batch size"):
        batch_size_of({"a": torch.zeros(2, 3), "b": torch.zeros(3, 3)})
    with pytest.raises(ValueError, match="0-dim"):
        batch_size_of({"a": torch.tensor(1.0)})


def test_slice_batch_int_keeps_batch_dim():
    s = _nested(3)
    row = slice_batch(s, 1)
    assert row["h"].shape == (1, 2, 4)
    assert torch.equal(row["h"][0], s["h"][1])
    last = slice_batch(s, -1)
    assert torch.equal(last["h"][0], s["h"][2])
    assert slice_batch(None, 0) is None


def test_slice_batch_sequence_and_tensor_index():
    s = _nested(4)
    sub = slice_batch(s, [3, 0])
    assert sub["h"].shape == (2, 2, 4)
    assert torch.equal(sub["h"][0], s["h"][3])
    sub_t = slice_batch(s, torch.tensor([1, 2]))
    assert torch.equal(sub_t["pair"].a, s["pair"].a[1:3])
    sub_mask = slice_batch(s, torch.tensor([True, False, True, False]))
    assert sub_mask["h"].shape[0] == 2


def test_cat_batch_roundtrip_of_rows():
    s = _nested(4)
    rows = [slice_batch(s, i) for i in range(4)]
    _assert_tree_equal(cat_batch(rows), s)
    assert isinstance(cat_batch(rows)["pair"], Pair)


def test_cat_batch_none_and_mismatch():
    assert cat_batch([None, None]) is None
    with pytest.raises(ValueError):
        cat_batch([None, {"h": torch.zeros(1, 2)}])
    with pytest.raises(ValueError):
        cat_batch([{"h": torch.zeros(1, 2)}, {"c": torch.zeros(1, 2)}])
    with pytest.raises(ValueError):
        cat_batch([])


def test_where_done_replaces_only_done_rows():
    state = {"mem": torch.full((3, 2), 5.0), "len": torch.tensor([4, 4, 4])}
    reset = {"mem": torch.zeros(3, 2), "len": torch.zeros(3, dtype=torch.long)}
    out = where_done(torch.tensor([False, True, False]), reset, state)
    assert torch.equal(out["mem"][1], torch.zeros(2))
    assert torch.equal(out["mem"][0], torch.full((2,), 5.0))
    assert out["len"].tolist() == [4, 0, 4]
    assert out["len"].dtype == torch.long
    assert where_done(torch.tensor([True]), None, None) is None


def test_numpy_roundtrip_preserves_structure_dtype_values():
    s = _nested(2)
    payload = state_to_numpy(s)
    assert isinstance(payload["h"], np.ndarray)
    assert isinstance(payload["pair"], Pair)
    assert payload["pair"].b[0].dtype == np.int64
    assert payload["none"] is None
    back = state_from_numpy(payload)
    _assert_tree_equal(back, s)
    assert state_to_numpy(None) is None
    assert state_from_numpy(None) is None


def test_state_to_numpy_is_a_copy():
    t = torch.zeros(2, 2)
    payload = state_to_numpy({"x": t})
    t.add_(1.0)
    assert payload["x"].sum() == 0.0


def test_state_to_device():
    s = _nested(2)
    moved = state_to(s, "cpu")
    _assert_tree_equal(moved, s)
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_state.py -v`

Expected: `ModuleNotFoundError: No module named 'colosseum.networks.state'`.

- [ ] **Step 3: Implement `src/colosseum/networks/state.py`**

```python
"""Opaque model-state pytrees.

A ``State`` is ``None``, a ``torch.Tensor``, or a (possibly nested) ``tuple``,
``list`` or ``dict`` of those. Every tensor leaf has the batch dimension first
(``dim 0``). Models decide what their state contains; the framework only
slices, concatenates, resets, moves and serializes it with the helpers below.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

State = Any  # None | Tensor | tuple | list | dict[str, State]; tensor leaves are batch-first


def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def tree_map(fn: Callable[[Tensor], Tensor], state: State) -> State:
    """Apply ``fn`` to every tensor leaf; keep the container structure."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return fn(state)
    if _is_namedtuple(state):
        return type(state)(*(tree_map(fn, s) for s in state))
    if isinstance(state, tuple):
        return tuple(tree_map(fn, s) for s in state)
    if isinstance(state, list):
        return [tree_map(fn, s) for s in state]
    if isinstance(state, dict):
        return {k: tree_map(fn, v) for k, v in state.items()}
    raise TypeError(f"Unsupported state node type: {type(state).__name__}")


def _map2(fn: Callable[[Tensor, Tensor], Tensor], a: State, b: State) -> State:
    """``tree_map`` over two states with identical structure."""
    if a is None and b is None:
        return None
    if isinstance(a, Tensor) and isinstance(b, Tensor):
        return fn(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        mapped = [_map2(fn, x, y) for x, y in zip(a, b)]
        return type(a)(*mapped) if _is_namedtuple(a) else tuple(mapped)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [_map2(fn, x, y) for x, y in zip(a, b)]
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return {k: _map2(fn, a[k], b[k]) for k in a}
    raise ValueError(f"State structures differ: {type(a).__name__} vs {type(b).__name__}")


def tree_leaves(state: State) -> list[Tensor]:
    """All tensor leaves in deterministic (insertion) order."""
    leaves: list[Tensor] = []
    tree_map(lambda t: leaves.append(t) or t, state)
    return leaves


def batch_size_of(state: State) -> int | None:
    """Common batch size of all leaves, or None if the state has no leaves."""
    leaves = tree_leaves(state)
    if not leaves:
        return None
    sizes = set()
    for leaf in leaves:
        if leaf.dim() == 0:
            raise ValueError("State leaves must have a batch dimension (got a 0-dim tensor)")
        sizes.add(int(leaf.shape[0]))
    if len(sizes) != 1:
        raise ValueError(f"State leaves disagree on batch size: {sorted(sizes)}")
    return sizes.pop()


def slice_batch(state: State, idx: int | Sequence[int] | Tensor) -> State:
    """Select batch rows. An ``int`` index keeps the batch dim (result has batch 1)."""
    if isinstance(idx, int):
        return tree_map(lambda t: t[idx].unsqueeze(0), state)
    if isinstance(idx, Tensor):
        index = idx if idx.dtype == torch.bool else idx.long()
    else:
        index = torch.as_tensor(list(idx), dtype=torch.long)
    return tree_map(lambda t: t[index.to(t.device)], state)


def cat_batch(states: Sequence[State]) -> State:
    """Concatenate states along the batch dim. All-``None`` gives ``None``."""
    states = list(states)
    if not states:
        raise ValueError("cat_batch needs at least one state")
    first = states[0]
    if first is None:
        if any(s is not None for s in states):
            raise ValueError("Cannot concatenate None with non-None states")
        return None
    if isinstance(first, Tensor):
        if not all(isinstance(s, Tensor) for s in states):
            raise ValueError("State structures differ: expected tensors")
        return torch.cat(states, dim=0)
    if isinstance(first, tuple):
        if not all(isinstance(s, tuple) and len(s) == len(first) for s in states):
            raise ValueError("State structures differ: tuple length/type mismatch")
        parts = [cat_batch([s[i] for s in states]) for i in range(len(first))]
        return type(first)(*parts) if _is_namedtuple(first) else tuple(parts)
    if isinstance(first, list):
        if not all(isinstance(s, list) and len(s) == len(first) for s in states):
            raise ValueError("State structures differ: list length/type mismatch")
        return [cat_batch([s[i] for s in states]) for i in range(len(first))]
    if isinstance(first, dict):
        if not all(isinstance(s, dict) and s.keys() == first.keys() for s in states):
            raise ValueError("State structures differ: dict keys mismatch")
        return {k: cat_batch([s[k] for s in states]) for k in first}
    raise TypeError(f"Unsupported state node type: {type(first).__name__}")


def where_done(done: Tensor, reset: State, state: State) -> State:
    """Row-wise select: rows where ``done`` is True come from ``reset``, others from ``state``.

    ``done`` is a ``[B]`` bool tensor; ``reset`` and ``state`` share structure and batch size.
    """
    def _select(r: Tensor, s: Tensor) -> Tensor:
        mask = done.to(device=s.device, dtype=torch.bool).view(-1, *([1] * (s.dim() - 1)))
        return torch.where(mask, r.to(dtype=s.dtype, device=s.device), s)

    return _map2(_select, reset, state)


def state_to_numpy(state: State) -> Any:
    """Same structure with ``np.ndarray`` leaves (detached CPU copies)."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach().cpu().numpy().copy()
    if _is_namedtuple(state):
        return type(state)(*(state_to_numpy(s) for s in state))
    if isinstance(state, tuple):
        return tuple(state_to_numpy(s) for s in state)
    if isinstance(state, list):
        return [state_to_numpy(s) for s in state]
    if isinstance(state, dict):
        return {k: state_to_numpy(v) for k, v in state.items()}
    raise TypeError(f"Unsupported state node type: {type(state).__name__}")


def state_from_numpy(obj: Any, device: str | torch.device = "cpu") -> State:
    """Inverse of :func:`state_to_numpy`."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(obj)).to(device)
    if _is_namedtuple(obj):
        return type(obj)(*(state_from_numpy(o, device) for o in obj))
    if isinstance(obj, tuple):
        return tuple(state_from_numpy(o, device) for o in obj)
    if isinstance(obj, list):
        return [state_from_numpy(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: state_from_numpy(v, device) for k, v in obj.items()}
    raise TypeError(f"Unsupported state payload node type: {type(obj).__name__}")


def state_to(state: State, device: str | torch.device) -> State:
    """Move every leaf to ``device``."""
    return tree_map(lambda t: t.to(device), state)
```

- [ ] **Step 4: Run the tests again**

Run: `.venv/bin/python -m pytest tests/unit/test_state.py -v`

Expected: 12 passed.

- [ ] **Step 5: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!`; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/networks/state.py tests/unit/test_state.py
git commit -m "feat: add State pytree utilities for opaque model state"
```

---

### Task T1.2: `PolicyModel` protocol + `act()`

**Files:**
- Create: `src/colosseum/networks/model.py`
- Modify: `src/colosseum/networks/distributions.py` (add `Distribution.cat` and implementations; `DiagGaussianDist` keeps `_mean`/`_log_std`)
- Test: `tests/unit/test_model.py`

**Interfaces:**
- Consumes: T1.1 (`State`, `batch_size_of`, `tree_leaves`, `where_done`).
- Produces:
  - Contract `colosseum.networks.model`: `StepOutput(dist, value, state)`, `UnrollOutput(dist, value)`, `ActOutput(actions, log_probs, values, state)`, `PolicyModel` (`initial_state(batch_size, device="cpu") -> State` default `None`; abstract `step(obs, state, action_mask=None) -> StepOutput` — the model applies the mask; `unroll(obs, state0, dones, action_mask=None) -> UnrollOutput`; `reset_state(state, done)`; `update_normalizers(obs)` no-op; property `is_stateful`), `act(model, obs, state, action_mask=None, deterministic=False) -> ActOutput` (runs under `torch.no_grad()`).
  - Default `unroll` semantics: a loop over `step()` with `state = reset_state(out.state, dones[t])` after every step, distributions joined with `Distribution.cat`, outputs time-major `[T*B]`. A stateless model (`state0 is None and not is_stateful`) is evaluated in one batched `step` on `[T*B, ...]` — equivalent and faster.
  - `Distribution.cat(dists: Sequence[Distribution]) -> Distribution` (classmethod; base raises `NotImplementedError`), implemented for `CategoricalDist`, `DiagGaussianDist`, `CompositeDist`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_model.py`:

```python
"""Unit tests for the PolicyModel protocol, act() and Distribution.cat."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from colosseum.networks.distributions import CategoricalDist, CompositeDist, DiagGaussianDist
from colosseum.networks.model import ActOutput, PolicyModel, StepOutput, UnrollOutput, act

OBS, A = 3, 4


class CounterModel(PolicyModel):
    """Stateful toy model: the state counts steps since the episode start."""

    def __init__(self) -> None:
        super().__init__()
        self.pi = nn.Linear(OBS + 1, A)
        self.v = nn.Linear(OBS + 1, 1)

    def initial_state(self, batch_size, device="cpu"):
        return {"count": torch.zeros(batch_size, 1, device=device)}

    def step(self, obs, state, action_mask=None):
        x = torch.cat([obs, state["count"]], dim=-1)
        dist = CategoricalDist(self.pi(x))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, self.v(x).squeeze(-1), {"count": state["count"] + 1.0})


class StatelessModel(PolicyModel):
    def __init__(self) -> None:
        super().__init__()
        self.pi = nn.Linear(OBS, A)
        self.v = nn.Linear(OBS, 1)

    def step(self, obs, state, action_mask=None):
        dist = CategoricalDist(self.pi(obs))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, self.v(obs).squeeze(-1), None)


def _manual_unroll(model, obs, dones, masks=None):
    T, B = obs.shape[:2]
    state = model.initial_state(B)
    logits, values = [], []
    for t in range(T):
        out = model.step(obs[t], state, None if masks is None else masks[t])
        logits.append(out.dist.logits)
        values.append(out.value)
        state = out.state
        if state is not None:
            keep = (~dones[t]).float().unsqueeze(-1)
            state = {"count": state["count"] * keep}
    return torch.cat(logits), torch.cat(values)


@pytest.mark.parametrize("model_cls", [CounterModel, StatelessModel])
def test_default_unroll_matches_step_loop_with_resets(model_cls):
    torch.manual_seed(0)
    model = model_cls()
    T, B = 5, 3
    obs = torch.randn(T, B, OBS)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1, 0] = True
    dones[3, 2] = True
    masks = torch.ones(T, B, A, dtype=torch.bool)
    masks[:, :, 0] = False

    out = model.unroll(obs, model.initial_state(B), dones, masks)
    assert isinstance(out, UnrollOutput)
    assert out.value.shape == (T * B,)
    ref_logits, ref_values = _manual_unroll(model, obs, dones, masks)
    assert torch.allclose(out.dist.logits, ref_logits, atol=1e-6)
    assert torch.allclose(out.value, ref_values, atol=1e-6)
    actions = torch.randint(1, A, (T * B,))
    assert out.dist.log_prob(actions).shape == (T * B,)


def test_unroll_is_time_major():
    torch.manual_seed(0)
    model = CounterModel()
    T, B = 3, 2
    obs = torch.randn(T, B, OBS)
    out = model.unroll(obs, model.initial_state(B), torch.zeros(T, B, dtype=torch.bool))
    state = model.initial_state(B)
    for t in range(T):
        step = model.step(obs[t], state)
        state = step.state
        for b in range(B):
            assert torch.allclose(out.value[t * B + b], step.value[b], atol=1e-6)


def test_reset_state_resets_only_done_rows():
    model = CounterModel()
    state = {"count": torch.tensor([[3.0], [5.0]])}
    out = model.reset_state(state, torch.tensor([True, False]))
    assert out["count"].tolist() == [[0.0], [5.0]]
    assert model.reset_state(None, torch.tensor([True])) is None


def test_is_stateful_and_default_hooks():
    assert CounterModel().is_stateful
    stateless = StatelessModel()
    assert not stateless.is_stateful
    assert stateless.initial_state(4) is None
    assert stateless.update_normalizers(torch.zeros(2, OBS)) is None


def test_act_samples_masked_actions_and_returns_state():
    torch.manual_seed(0)
    model = CounterModel()
    obs = torch.randn(6, OBS)
    mask = torch.zeros(6, A, dtype=torch.bool)
    mask[:, 2] = True
    out = act(model, obs, model.initial_state(6), mask)
    assert isinstance(out, ActOutput)
    assert out.actions.tolist() == [2] * 6
    assert out.log_probs.shape == (6,) and torch.allclose(out.log_probs, torch.zeros(6), atol=1e-6)
    assert out.values.shape == (6,)
    assert out.state["count"].tolist() == [[1.0]] * 6
    assert not out.log_probs.requires_grad


def test_act_deterministic_takes_mode():
    torch.manual_seed(0)
    model = StatelessModel()
    obs = torch.randn(5, OBS)
    out = act(model, obs, None, deterministic=True)
    expected = model.pi(obs).argmax(dim=-1)
    assert torch.equal(out.actions, expected)
    assert out.state is None


def test_distribution_cat_categorical_gaussian_composite():
    torch.manual_seed(0)
    c1, c2 = CategoricalDist(torch.randn(2, A)), CategoricalDist(torch.randn(3, A))
    cc = CategoricalDist.cat([c1, c2])
    a = torch.randint(0, A, (5,))
    assert torch.allclose(cc.log_prob(a), torch.cat([c1.log_prob(a[:2]), c2.log_prob(a[2:])]))

    g1 = DiagGaussianDist(torch.randn(2, 2), torch.zeros(1, 2).expand(2, -1))
    g2 = DiagGaussianDist(torch.randn(1, 2), torch.zeros(1, 2))
    gc = DiagGaussianDist.cat([g1, g2])
    x = torch.randn(3, 2)
    assert torch.allclose(gc.log_prob(x), torch.cat([g1.log_prob(x[:2]), g2.log_prob(x[2:])]))

    k1 = CompositeDist({"d": c1, "s": DiagGaussianDist(torch.randn(2, 1), torch.zeros(2, 1))})
    k2 = CompositeDist({"d": c2, "s": DiagGaussianDist(torch.randn(3, 1), torch.zeros(3, 1))})
    kc = CompositeDist.cat([k1, k2])
    flat = torch.cat([a.float().unsqueeze(-1), torch.randn(5, 1)], dim=-1)
    assert torch.allclose(kc.log_prob(flat), torch.cat([k1.log_prob(flat[:2]), k2.log_prob(flat[2:])]))


def test_masked_categorical_cat_keeps_mask():
    logits = torch.zeros(2, A)
    mask = torch.tensor([[True, False, True, False], [False, True, True, True]])
    d = CategoricalDist(logits, mask=mask)
    cc = CategoricalDist.cat([d, d])
    assert torch.isinf(cc.logits[0, 1]) and torch.isinf(cc.logits[2, 1])
    assert torch.isfinite(cc.entropy()).all()
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_model.py -v`

Expected: `ModuleNotFoundError: No module named 'colosseum.networks.model'`.

- [ ] **Step 3: Add `cat` to the distributions**

In `src/colosseum/networks/distributions.py`:

1. Replace:

```python
from abc import ABC, abstractmethod

import torch
```

with:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
```

2. Replace:

```python
        Default: no-op (returns self). Override for discrete distributions.
        """
        return self
```

with:

```python
        Default: no-op (returns self). Override for discrete distributions.
        """
        return self

    @classmethod
    def cat(cls, dists: Sequence[Distribution]) -> Distribution:
        """Concatenate same-type distributions along the batch dimension.

        Used by the default ``PolicyModel.unroll`` (a per-step loop). Custom
        distributions must override it to be used with that default.
        """
        raise NotImplementedError(f"{cls.__name__}.cat is not implemented")
```

3. Replace:

```python
        return CategoricalDist(logits=self.logits, mask=mask)
```

with:

```python
        return CategoricalDist(logits=self.logits, mask=mask)

    @classmethod
    def cat(cls, dists: Sequence[CategoricalDist]) -> CategoricalDist:
        # logits are already normalized and masked (-inf), so no mask is needed.
        return CategoricalDist(logits=torch.cat([d.logits for d in dists], dim=0))
```

4. Replace:

```python
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self._dist = torch.distributions.Normal(mean, log_std.exp())
```

with:

```python
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self._mean = mean
        self._log_std = log_std
        self._dist = torch.distributions.Normal(mean, log_std.exp())
```

5. Replace:

```python
        return torch.distributions.kl_divergence(self._dist, other._dist).sum(dim=-1)
```

with:

```python
        return torch.distributions.kl_divergence(self._dist, other._dist).sum(dim=-1)

    @classmethod
    def cat(cls, dists: Sequence[DiagGaussianDist]) -> DiagGaussianDist:
        return DiagGaussianDist(
            torch.cat([d._mean for d in dists], dim=0),
            torch.cat([d._log_std for d in dists], dim=0),
        )
```

6. Append to the end of the file (inside `class CompositeDist`, after `kl_divergence`):

```python

    @classmethod
    def cat(cls, dists: Sequence[CompositeDist]) -> CompositeDist:
        keys = dists[0]._keys
        for d in dists:
            if d._keys != keys:
                raise ValueError(f"Key mismatch: {d._keys} vs {keys}")
        return CompositeDist({k: type(dists[0]._dists[k]).cat([d._dists[k] for d in dists]) for k in keys})
```

- [ ] **Step 4: Implement `src/colosseum/networks/model.py`**

```python
"""PolicyModel: the stateful actor-critic protocol used by workers, learners and eval.

A model maps ``(obs, state)`` to an action distribution, a value and the next
state. ``state`` is an opaque pytree (see :mod:`colosseum.networks.state`);
stateless models (MLP, CNN) use ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.networks.distributions import Distribution
from colosseum.networks.state import State, batch_size_of, tree_leaves, where_done


class StepOutput(NamedTuple):
    dist: Distribution  # batch [B]
    value: Tensor  # [B]
    state: State


class UnrollOutput(NamedTuple):
    dist: Distribution  # batch [T*B], time-major flatten (index t*B + b)
    value: Tensor  # [T*B], time-major flatten


class ActOutput(NamedTuple):
    actions: Tensor  # [B, *action_shape] (flat action layout from ActionSpec)
    log_probs: Tensor  # [B]
    values: Tensor  # [B]
    state: State


class PolicyModel(nn.Module, ABC):
    """Actor-critic with an optional recurrent/memory state."""

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        """State at the start of an episode. Stateless models return None."""
        return None

    @abstractmethod
    def step(self, obs: Tensor, state: State, action_mask: Tensor | None = None) -> StepOutput:
        """One timestep for a batch: obs ``[B, *obs_shape]`` -> (dist [B], value [B], next state).

        ``action_mask`` (``[B, mask_size]`` bool) must be applied to the returned dist.
        """

    def unroll(
        self,
        obs: Tensor,
        state0: State,
        dones: Tensor,
        action_mask: Tensor | None = None,
    ) -> UnrollOutput:
        """Process sequences: obs ``[T, B, ...]``, dones ``[T, B]`` (``dones[t]``: the
        episode ended AFTER step t, so the state is reset before step t+1),
        mask ``[T, B, A]`` or None. Returns time-major flattened ``[T*B]`` outputs.

        Default: a Python loop over :meth:`step` with :meth:`reset_state` after
        every step, which is correct for any model. Stateless models are
        evaluated in one batched ``step`` (equivalent, faster). Distributions
        are concatenated with ``Distribution.cat``.
        """
        T, B = obs.shape[0], obs.shape[1]
        if state0 is None and not self.is_stateful:
            flat_mask = None if action_mask is None else action_mask.reshape(T * B, *action_mask.shape[2:])
            out = self.step(obs.reshape(T * B, *obs.shape[2:]), None, flat_mask)
            return UnrollOutput(dist=out.dist, value=out.value)

        dists: list[Distribution] = []
        values: list[Tensor] = []
        state = state0
        for t in range(T):
            out = self.step(obs[t], state, None if action_mask is None else action_mask[t])
            dists.append(out.dist)
            values.append(out.value)
            state = self.reset_state(out.state, dones[t])
        return UnrollOutput(dist=type(dists[0]).cat(dists), value=torch.cat(values, dim=0))

    def reset_state(self, state: State, done: Tensor) -> State:
        """Replace the rows of ``state`` where ``done`` ([B] bool) with initial-state rows."""
        if state is None:
            return None
        batch = batch_size_of(state)
        device = tree_leaves(state)[0].device
        return where_done(done, self.initial_state(batch, device), state)

    def update_normalizers(self, obs: Tensor) -> None:
        """Update running observation statistics (no-op unless the model has any)."""
        return None

    @property
    def is_stateful(self) -> bool:
        return self.initial_state(1) is not None


@torch.no_grad()
def act(
    model: PolicyModel,
    obs: Tensor,
    state: State,
    action_mask: Tensor | None = None,
    deterministic: bool = False,
) -> ActOutput:
    """Inference helper: step the model, then sample (or take the mode) and score the action."""
    out = model.step(obs, state, action_mask)
    actions = out.dist.mode() if deterministic else out.dist.sample()
    log_probs = out.dist.log_prob(actions)
    return ActOutput(actions=actions, log_probs=log_probs, values=out.value, state=out.state)
```

- [ ] **Step 5: Run the tests again**

Run: `.venv/bin/python -m pytest tests/unit/test_model.py tests/unit/test_distributions.py tests/unit/test_composite_actions.py -v`

Expected: all pass (`test_model.py`: 9 tests).

- [ ] **Step 6: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!`; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/colosseum/networks/model.py src/colosseum/networks/distributions.py tests/unit/test_model.py
git commit -m "feat: add PolicyModel protocol, act() and Distribution.cat"
```

---

### Task T1.3: Cores: NoCore, LSTMCore, GRUCore, WindowAttentionCore

**Files:**
- Create: `src/colosseum/networks/cores.py`
- Test: `tests/unit/test_cores.py`

**Interfaces:**
- Consumes: T1.1 (`State`, `batch_size_of`, `tree_leaves`, `where_done`).
- Produces (contract `colosseum.networks.cores`):
  - `Core(nn.Module, ABC)` with attributes `input_dim`, `output_dim`; `initial_state(batch_size, device="cpu") -> State` (default `None`); abstract `step(x [B, input_dim], state) -> (features [B, output_dim], state)`; `unroll(x [T, B, input_dim], state0, dones [T, B]) -> [T, B, output_dim]` (reference loop: `step`, then `reset_state(state, dones[t])`); `reset_state(state, done)` (rows where `done` take the initial state).
  - `NoCore(input_dim)` — identity, state `None`, `unroll` returns `x`.
  - `LSTMCore(input_dim, hidden_size=128, num_layers=1)` — submodule `rnn` (`nn.LSTM`, sequence-first); state `{"h": [B, L, H], "c": [B, L, H]}`, transposed to `nn.LSTM` layout inside `step`.
  - `GRUCore(input_dim, hidden_size=128, num_layers=1)` — submodule `rnn` (`nn.GRU`); state `{"h": [B, L, H]}`.
  - `WindowAttentionCore(input_dim, d_model=64, window=16, num_heads=4, num_layers=1)` — state `{"mem": [B, window, d_model] float, "len": [B] long}`. `mem` holds the projected latents of the previous steps, right-aligned (newest at `mem[:, -1]`); only the last `len` slots are valid. Each step runs `num_layers` pre-LN attention blocks over `[mem; current]` with a causal mask where a position may attend to valid positions and to itself, adds a learned positional embedding (`pos`, `[window+1, d_model]`), and returns the normalized output at the current position. `len` saturates at `window`; reset zeroes `mem` and `len`. Raises `ValueError` if `window < 1` or `d_model % num_heads != 0`. In SP1 all cores use the reference per-step `unroll`; fast sequence paths are SP6.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cores.py`:

```python
"""Unit tests for colosseum.networks.cores."""

from __future__ import annotations

import pytest
import torch

from colosseum.networks.cores import GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.state import batch_size_of, cat_batch, slice_batch, tree_leaves

IN = 6


def _make(kind: str):
    if kind == "none":
        return NoCore(IN)
    if kind == "lstm":
        return LSTMCore(IN, hidden_size=10, num_layers=2)
    if kind == "gru":
        return GRUCore(IN, hidden_size=10, num_layers=1)
    if kind == "window":
        return WindowAttentionCore(IN, d_model=8, window=3, num_heads=2, num_layers=2)
    raise ValueError(kind)


KINDS = ["none", "lstm", "gru", "window"]


@pytest.mark.parametrize("kind", KINDS)
def test_output_dim_and_state_batch_dims(kind):
    core = _make(kind)
    state = core.initial_state(4)
    y, new_state = core.step(torch.randn(4, IN), state)
    assert y.shape == (4, core.output_dim)
    if kind == "none":
        assert state is None and new_state is None
    else:
        assert batch_size_of(state) == 4 and batch_size_of(new_state) == 4


@pytest.mark.parametrize("kind", KINDS)
def test_unroll_equals_step_loop_with_resets(kind):
    torch.manual_seed(0)
    core = _make(kind)
    T, B = 7, 3
    x = torch.randn(T, B, IN)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[2, 0] = True
    dones[4, 1] = True
    dones[5, 0] = True

    state = core.initial_state(B)
    ref = []
    for t in range(T):
        y, state = core.step(x[t], state)
        ref.append(y)
        state = core.reset_state(state, dones[t])
    out = core.unroll(x, core.initial_state(B), dones)
    assert out.shape == (T, B, core.output_dim)
    assert torch.allclose(out, torch.stack(ref), atol=1e-6)


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_reset_state_replaces_only_done_rows(kind):
    torch.manual_seed(0)
    core = _make(kind)
    state = core.initial_state(3)
    for _ in range(4):
        _, state = core.step(torch.randn(3, IN), state)
    reset = core.reset_state(state, torch.tensor([False, True, False]))
    init = core.initial_state(1)
    for a, b, i in zip(tree_leaves(reset), tree_leaves(state), tree_leaves(init)):
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
        assert torch.equal(a[1], i[0])


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_reset_isolates_post_done_steps(kind):
    """After a done at t=1, outputs at t>=2 must not depend on inputs at t<=1."""
    torch.manual_seed(0)
    core = _make(kind)
    T, B = 5, 1
    xa = torch.randn(T, B, IN)
    xb = xa.clone()
    xb[0] = torch.randn(B, IN)
    xb[1] = torch.randn(B, IN)
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[1, 0] = True
    ya = core.unroll(xa, core.initial_state(B), dones)
    yb = core.unroll(xb, core.initial_state(B), dones)
    assert torch.allclose(ya[2:], yb[2:], atol=1e-6)
    assert not torch.allclose(ya[:2], yb[:2], atol=1e-6)


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_batched_step_equals_per_row_step(kind):
    """Grouped inference: cat_batch/slice_batch around step must not mix rows."""
    torch.manual_seed(0)
    core = _make(kind)
    rows = []
    for _ in range(3):
        s = core.initial_state(1)
        _, s = core.step(torch.randn(1, IN), s)
        rows.append(s)
    x = torch.randn(3, IN)
    y_batch, s_batch = core.step(x, cat_batch(rows))
    for i in range(3):
        y_i, s_i = core.step(x[i:i + 1], rows[i])
        assert torch.allclose(y_batch[i:i + 1], y_i, atol=1e-6)
        for a, b in zip(tree_leaves(slice_batch(s_batch, i)), tree_leaves(s_i)):
            assert torch.allclose(a.float(), b.float(), atol=1e-6)


def test_window_attention_ignores_masked_memory():
    torch.manual_seed(0)
    core = _make("window")
    x = torch.randn(2, IN)
    clean = core.initial_state(2)
    dirty = {"mem": torch.randn_like(clean["mem"]) * 100.0, "len": clean["len"].clone()}
    y_clean, _ = core.step(x, clean)
    y_dirty, _ = core.step(x, dirty)
    assert torch.allclose(y_clean, y_dirty, atol=1e-6)


def test_window_attention_sees_only_last_window_steps():
    torch.manual_seed(0)
    core = _make("window")  # window=3
    T = 6
    xa = torch.randn(T, 1, IN)
    xb = xa.clone()
    xb[0] = torch.randn(1, IN)  # differs only at t=0
    dones = torch.zeros(T, 1, dtype=torch.bool)
    ya = core.unroll(xa, core.initial_state(1), dones)
    yb = core.unroll(xb, core.initial_state(1), dones)
    assert not torch.allclose(ya[3], yb[3], atol=1e-6)  # t=3 still attends to t=0
    assert torch.allclose(ya[4:], yb[4:], atol=1e-6)  # t>=4: t=0 has left the window


def test_window_attention_len_saturates_and_reset_clears():
    core = _make("window")
    state = core.initial_state(2)
    for _ in range(5):
        _, state = core.step(torch.randn(2, IN), state)
    assert state["len"].tolist() == [3, 3]
    reset = core.reset_state(state, torch.tensor([True, False]))
    assert reset["len"].tolist() == [0, 3]
    assert torch.count_nonzero(reset["mem"][0]) == 0


@pytest.mark.parametrize("kind", ["lstm", "gru", "window"])
def test_unroll_backpropagates_into_core_parameters(kind):
    torch.manual_seed(0)
    core = _make(kind)
    out = core.unroll(torch.randn(4, 2, IN), core.initial_state(2), torch.zeros(4, 2, dtype=torch.bool))
    out.sum().backward()
    assert all(p.grad is not None for p in core.parameters())


def test_window_attention_rejects_bad_sizes():
    with pytest.raises(ValueError):
        WindowAttentionCore(IN, d_model=10, num_heads=4)
    with pytest.raises(ValueError):
        WindowAttentionCore(IN, window=0)
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_cores.py -v`

Expected: `ModuleNotFoundError: No module named 'colosseum.networks.cores'`.

- [ ] **Step 3: Implement `src/colosseum/networks/cores.py`**

The window-attention mask is built so that no attention row is fully masked (every position may attend to itself), which keeps padded memory slots from producing NaNs; valid positions never attend to padded ones, so the padding contents do not matter (see `test_window_attention_ignores_masked_memory`).

```python
"""Cores: the (optionally stateful) trunk between the encoder and the heads.

``ComposedModel`` runs ``encoder -> core -> heads``. A core turns a latent
``[B, input_dim]`` plus its state into features ``[B, output_dim]`` and the
next state. State leaves are batch-first (see :mod:`colosseum.networks.state`).

In SP1 every core's ``unroll`` is the reference per-step loop; fast sequence
paths (cuDNN RNN, full-sequence attention masks) come later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.networks.state import State, batch_size_of, tree_leaves, where_done


class Core(nn.Module, ABC):
    input_dim: int
    output_dim: int

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return None

    @abstractmethod
    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        """``x`` [B, input_dim] -> (features [B, output_dim], next state)."""

    def unroll(self, x: Tensor, state0: State, dones: Tensor) -> Tensor:
        """``x`` [T, B, input_dim] -> [T, B, output_dim]; state reset after step t where ``dones[t]``."""
        state = state0
        outputs = []
        for t in range(x.shape[0]):
            y, state = self.step(x[t], state)
            outputs.append(y)
            state = self.reset_state(state, dones[t])
        return torch.stack(outputs, dim=0)

    def reset_state(self, state: State, done: Tensor) -> State:
        """Rows where ``done`` ([B]) is True are replaced by initial-state rows."""
        if state is None:
            return None
        batch = batch_size_of(state)
        device = tree_leaves(state)[0].device
        return where_done(done, self.initial_state(batch, device), state)


class NoCore(Core):
    """Identity core for stateless models."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        return x, None

    def unroll(self, x: Tensor, state0: State, dones: Tensor) -> Tensor:
        return x


class LSTMCore(Core):
    """LSTM trunk. State ``{"h": [B, L, H], "c": [B, L, H]}``."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        zeros = torch.zeros(batch_size, self.num_layers, self.hidden_size, device=device)
        return {"h": zeros, "c": zeros.clone()}

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        h = state["h"].transpose(0, 1).contiguous()
        c = state["c"].transpose(0, 1).contiguous()
        y, (h, c) = self.rnn(x.unsqueeze(0), (h, c))
        return y.squeeze(0), {"h": h.transpose(0, 1).contiguous(), "c": c.transpose(0, 1).contiguous()}


class GRUCore(Core):
    """GRU trunk. State ``{"h": [B, L, H]}``."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return {"h": torch.zeros(batch_size, self.num_layers, self.hidden_size, device=device)}

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        h = state["h"].transpose(0, 1).contiguous()
        y, h = self.rnn(x.unsqueeze(0), h)
        return y.squeeze(0), {"h": h.transpose(0, 1).contiguous()}


class _AttentionBlock(nn.Module):
    """Pre-LayerNorm transformer block (no gating)."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, h: Tensor, blocked: Tensor) -> Tensor:
        a = self.ln1(h)
        h = h + self.attn(a, a, a, attn_mask=blocked, need_weights=False)[0]
        return h + self.ff(self.ln2(h))


class WindowAttentionCore(Core):
    """Causal attention of the current latent over the last ``window`` latents of the episode.

    State ``{"mem": [B, window, d_model] float, "len": [B] long}``: ``mem`` holds
    the projected latents of the previous steps, right-aligned (the newest is
    ``mem[:, -1]``); only the last ``len`` slots are valid. Each step re-runs
    the ``num_layers`` blocks over ``[valid memory; current]`` with a causal
    mask and returns the output at the current position (GTrXL-lite without
    gating). Resetting the state sets ``len`` to 0 and zeroes ``mem``.
    """

    def __init__(self, input_dim: int, d_model: int = 64, window: int = 16,
                 num_heads: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.input_dim = input_dim
        self.output_dim = d_model
        self.d_model = d_model
        self.window = window
        self.num_heads = num_heads
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(window + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(_AttentionBlock(d_model, num_heads) for _ in range(num_layers))
        self.out_norm = nn.LayerNorm(d_model)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return {
            "mem": torch.zeros(batch_size, self.window, self.d_model, device=device),
            "len": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        z = self.in_proj(x)  # [B, D]
        mem, length = state["mem"], state["len"]
        W = self.window
        seq = torch.cat([mem.to(z.dtype), z.unsqueeze(1)], dim=1) + self.pos  # [B, W+1, D]

        pos = torch.arange(W + 1, device=z.device)
        valid = pos.unsqueeze(0) >= (W - length).unsqueeze(1)  # [B, W+1]; current slot always valid
        causal = pos.unsqueeze(0) <= pos.unsqueeze(1)  # [W+1 (query), W+1 (key)]
        eye = torch.eye(W + 1, dtype=torch.bool, device=z.device)
        allowed = causal.unsqueeze(0) & (valid.unsqueeze(1) | eye.unsqueeze(0))  # every row keeps itself
        blocked = (~allowed).repeat_interleave(self.num_heads, dim=0)  # [B*heads, W+1, W+1]

        h = seq
        for block in self.blocks:
            h = block(h, blocked)
        out = self.out_norm(h[:, -1])

        new_mem = torch.cat([mem[:, 1:], z.unsqueeze(1).to(mem.dtype)], dim=1)
        new_len = torch.clamp(length + 1, max=W)
        return out, {"mem": new_mem, "len": new_len}
```

- [ ] **Step 4: Run the tests again**

Run: `.venv/bin/python -m pytest tests/unit/test_cores.py -v`

Expected: 24 passed.

- [ ] **Step 5: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!`; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/colosseum/networks/cores.py tests/unit/test_cores.py
git commit -m "feat: add Core protocol with NoCore, LSTMCore, GRUCore and WindowAttentionCore"
```

---

### Task T1.4: `ComposedModel`, `networks` schema, `build_model`, `validate_config`, errors module, migration of helpers/examples/configs

**Files:**
- Create: `src/colosseum/core/errors.py`, `src/colosseum/networks/composed.py`
- Modify: `src/colosseum/core/config.py` (`NetworkConfig` → `CoreConfig` + new `NetworkConfig`; pydantic import)
- Modify: `src/colosseum/core/registry.py` (rewrite: `build_model`, `validate_config`, transitional `build_network`)
- Modify: `tests/helpers.py` (rewrite: heads take `in_dim`, model helpers)
- Modify: `examples/tic_tac_toe/networks.py`, `examples/composite_action/networks.py`, `examples/space_miners/networks.py` (heads take `in_dim`)
- Modify: `configs/examples/{chase,space_miners,tic_tac_toe,tic_tac_toe_multi}.yaml` (add `core: null`)
- Modify: `tests/unit/test_recurrent.py` (the three `build_network` tests), `tests/unit/test_review_fixes.py` (`_make_recurrent_net` head kwargs)
- Test: `tests/unit/test_registry.py`

**Interfaces:**
- Consumes: T1.2 (`PolicyModel`, `StepOutput`, `UnrollOutput`), T1.3 (`Core`, `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`), T1.1 (`tree_leaves`).
- Produces:
  - Contract `colosseum.core.errors`: `ColosseumError`, `ConfigError(ColosseumError)`, `EnvContractError(ColosseumError)`.
  - Contract `colosseum.networks.composed.ComposedModel(encoder, core, policy_head, value_head)`. Submodule names: `encoder`, `core`, `policy`, `value` (a stateless model's `state_dict` has only `encoder.*`, `policy.*`, `value.*` keys — the same keys as the old `ActorCriticNetwork`). `step` = encoder → `core.step` → heads, mask via `dist.apply_mask`; `unroll` encodes all `T*B` observations at once, runs `core.unroll` and the heads once on `[T*B, O]`; `initial_state`/`reset_state` delegate to the core.
  - Contract `colosseum.core.config`: `CoreConfig` (`class_path` aliased `class`, `kwargs`; `extra="forbid"`, `populate_by_name=True`), `NetworkConfig` (`model_class`, `encoder_class`, `core`, `policy_class`, `value_class`, `kwargs`; `extra="forbid"`; validator: `model_class` alone, or all of encoder/policy/value). `recurrent_type`, `recurrent_hidden_size`, `recurrent_num_layers` are removed (unknown keys now fail validation).
  - Contract `colosseum.core.registry`: `build_model(config) -> PolicyModel`, `validate_config(config) -> None` (raises `ConfigError`). `validate_config` builds the env and the model, runs `step` on a batch of 2 copies of the env's first observation (with the env's `action_mask` from `reset` info, if any) and `unroll` on `[T=3, B=2]` with a reset in the middle, and checks: env/model construction, the policy returns a `Distribution`, value shape `[B]`, every state leaf batch-first, action shape `(B, *ActionSpec.action_shape)`, mask size `== ActionSpec.flat_mask_size`, unroll outputs `[T*B]`. Head/core dimension errors carry the hint `core.output_dim=<O>`.
  - Transitional `colosseum.core.registry.build_network(config) -> ActorCriticNetwork` on the new schema (`core` null/`NoCore` → feedforward, `LSTMCore`/`GRUCore` → `nn.LSTM`/`nn.GRU` with `hidden_size`/`num_layers` from `core.kwargs`); used only by callers that still need `ActorCriticNetwork` (training until T1.5, eval/BC CLI until T1.7); deleted in T1.7.
  - Example heads (`TicTacToePolicy`, `TicTacToeValue`, `ChasePolicy`, `ChaseValue`, `SpaceMinersPolicy`, `SpaceMinersValue`) accept `in_dim` (default = their old hard-wired size), so `build_model` sizes them to any core.
  - `tests/helpers.py` (in addition to T0.2/T0.5 names): `SimplePolicy(in_dim=16, num_actions=4)`, `SimpleValue(in_dim=16)` (the first parameter was `hidden_dim`), `FixedInputPolicy` (64 inputs, no `in_dim`), `BadShapeValue` (returns `[B, 1]`), `TinyMonolithicModel(obs_dim=27, num_actions=9)` (stateless `PolicyModel`), `CORE_KINDS = ("none", "lstm", "gru", "window")`, `make_core(kind, input_dim) -> Core` (LSTM/GRU `hidden_size=24`, window `d_model=16, window=3, num_heads=2` — sizes differ from the latent on purpose, R1-21), `make_simple_model(obs_dim=8, hidden_dim=16, num_actions=4, core="none") -> ComposedModel`, `make_ttt_model() -> ComposedModel` (tic-tac-toe example networks with `NoCore`); `make_simple_network` stays until T1.7.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_registry.py`:

```python
"""Tests for build_model / validate_config and the networks config schema."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from colosseum.core.config import ColosseumConfig, CoreConfig, EnvConfig, NetworkConfig, load_config
from colosseum.core.errors import ColosseumError, ConfigError
from colosseum.core.registry import build_model, validate_config
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.model import PolicyModel
from helpers import TinyMonolithicModel, example_config

TTT_ENV = "examples.tic_tac_toe.env.TicTacToeEnv"
TTT_NETS = "examples.tic_tac_toe.networks"


def _cfg(**networks) -> ColosseumConfig:
    return ColosseumConfig(env=EnvConfig(env_class=TTT_ENV), networks=NetworkConfig(**networks))


def _composed(core=None, policy=f"{TTT_NETS}.TicTacToePolicy", value=f"{TTT_NETS}.TicTacToeValue"):
    return _cfg(
        encoder_class=f"{TTT_NETS}.TicTacToeEncoder",
        core=core,
        policy_class=policy,
        value_class=value,
    )


def test_errors_hierarchy():
    assert issubclass(ConfigError, ColosseumError)


def test_core_config_accepts_class_alias_and_forbids_extra():
    cc = CoreConfig.model_validate({"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 8}})
    assert cc.class_path == "colosseum.networks.cores.LSTMCore"
    with pytest.raises(ValidationError):
        CoreConfig.model_validate({"class": "x.Y", "hidden_size": 8})


def test_network_config_requires_model_or_all_parts():
    with pytest.raises(ValidationError, match="missing: encoder_class, policy_class, value_class"):
        NetworkConfig()
    with pytest.raises(ValidationError, match="missing: value_class"):
        NetworkConfig(encoder_class="a.B", policy_class="a.C")
    with pytest.raises(ValidationError, match="must be omitted"):
        NetworkConfig(model_class="a.M", encoder_class="a.B")
    NetworkConfig(model_class="a.M")


def test_network_config_rejects_removed_recurrent_keys():
    with pytest.raises(ValidationError):
        NetworkConfig.model_validate({
            "encoder_class": "a.B", "policy_class": "a.C", "value_class": "a.D",
            "recurrent_type": "lstm",
        })


def test_agent_config_roundtrip_keeps_core():
    cfg = _composed(core={"class": "colosseum.networks.cores.GRUCore", "kwargs": {"hidden_size": 64}})
    again = cfg.get_agent_config("agent_0")
    assert again.networks.core.class_path == "colosseum.networks.cores.GRUCore"
    assert again.networks.core.kwargs == {"hidden_size": 64}


def test_build_model_default_is_composed_nocore():
    model = build_model(_composed())
    assert isinstance(model, ComposedModel)
    assert isinstance(model.core, NoCore)
    assert not model.is_stateful


def test_build_model_passes_in_dim_to_heads():
    model = build_model(_composed(core={"class": "colosseum.networks.cores.LSTMCore",
                                        "kwargs": {"hidden_size": 32}}))
    assert isinstance(model.core, LSTMCore)
    assert model.core.input_dim == 64
    assert model.policy.net.in_features == 32
    assert model.value.net[0].in_features == 32
    out = model.step(torch.zeros(2, 3, 3, 3), model.initial_state(2))
    assert out.value.shape == (2,)


def test_build_model_with_model_class():
    model = build_model(_cfg(model_class="helpers.TinyMonolithicModel"))
    assert isinstance(model, TinyMonolithicModel)
    with pytest.raises(ConfigError, match="must subclass colosseum.networks.model.PolicyModel"):
        build_model(_cfg(model_class="torch.nn.Linear", kwargs={"in_features": 2, "out_features": 2}))


def test_build_model_rejects_non_core_class():
    with pytest.raises(ConfigError, match="must subclass colosseum.networks.cores.Core"):
        build_model(_composed(core={"class": "torch.nn.Identity"}))


@pytest.mark.parametrize("name", ["tic_tac_toe.yaml", "tic_tac_toe_multi.yaml", "chase.yaml"])
def test_example_configs_build_and_validate(name):
    cfg = load_config(example_config(name))
    for aid in cfg.get_trainable_agent_ids():
        acfg = cfg.get_agent_config(aid)
        assert isinstance(build_model(acfg), PolicyModel)
        validate_config(acfg)


def test_validate_config_accepts_window_attention_core():
    validate_config(_composed(core={"class": "colosseum.networks.cores.WindowAttentionCore",
                                    "kwargs": {"d_model": 16, "window": 4, "num_heads": 2}}))
    model = build_model(_composed(core={"class": "colosseum.networks.cores.WindowAttentionCore"}))
    assert isinstance(model.core, WindowAttentionCore)


def test_validate_config_reports_head_dim_mismatch():
    cfg = _composed(
        core={"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}},
        policy="helpers.SimplePolicy",  # in_dim is passed -> fine
        value="helpers.SimpleValue",
    )
    validate_config(cfg)
    cfg_bad = _cfg(
        encoder_class=f"{TTT_NETS}.TicTacToeEncoder",
        core={"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}},
        policy_class="helpers.FixedInputPolicy",
        value_class=f"{TTT_NETS}.TicTacToeValue",
    )
    with pytest.raises(ConfigError, match="core.output_dim=32"):
        validate_config(cfg_bad)


def test_validate_config_reports_non_distribution_policy():
    with pytest.raises(ConfigError, match="must return a colosseum Distribution"):
        validate_config(_composed(policy="torch.nn.Identity"))


def test_validate_config_reports_value_shape():
    with pytest.raises(ConfigError, match=r"value must have shape \[B\]"):
        validate_config(_composed(value="helpers.BadShapeValue"))


def test_validate_config_reports_bad_env_and_bad_class():
    with pytest.raises(ConfigError, match="Failed to create env"):
        validate_config(ColosseumConfig(env=EnvConfig(env_class="nope.Env"),
                                        networks=NetworkConfig(model_class="helpers.TinyMonolithicModel")))
    with pytest.raises(ConfigError, match="Failed to build model"):
        validate_config(_composed(policy="nope.Policy"))
```

- [ ] **Step 2: Update the test helpers**

Replace the whole of `tests/helpers.py` with:

```python
"""Shared test helpers: toy environments and small networks/models."""

from __future__ import annotations

from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn

from colosseum.envs.base_env import BaseEnv
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.composed import ComposedModel
from colosseum.networks.cores import Core, GRUCore, LSTMCore, NoCore, WindowAttentionCore
from colosseum.networks.distributions import CategoricalDist
from colosseum.networks.model import PolicyModel, StepOutput

REPO_ROOT = Path(__file__).resolve().parent.parent


def example_config(name: str) -> Path:
    """Absolute path of ``configs/examples/<name>`` (tests run with cwd = tmp_path)."""
    return REPO_ROOT / "configs" / "examples" / name


# ---------------------------------------------------------------------------
# Small networks
# ---------------------------------------------------------------------------


class SimpleEncoder(BaseEncoder):
    def __init__(self, obs_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        self._latent_dim = hidden_dim
        self.fc = nn.Linear(obs_dim, hidden_dim)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, obs):
        return torch.relu(self.fc(obs))


class SimplePolicy(BasePolicy):
    def __init__(self, in_dim: int = 16, num_actions: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_actions)

    def forward(self, latent):
        return CategoricalDist(self.fc(latent))


class SimpleValue(BaseValue):
    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, latent):
        return self.fc(latent).squeeze(-1)


class FixedInputPolicy(BasePolicy):
    """Policy head hard-wired to 64 input features (no ``in_dim``): mismatches other cores."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 9)

    def forward(self, latent):
        return CategoricalDist(self.fc(latent))


class BadShapeValue(BaseValue):
    """Value head that forgets to squeeze: returns [B, 1] (validate_config must reject it)."""

    def __init__(self, in_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, latent):
        return self.fc(latent)


class TinyMonolithicModel(PolicyModel):
    """Stateless PolicyModel used through ``networks.model_class``."""

    def __init__(self, obs_dim: int = 27, num_actions: int = 9) -> None:
        super().__init__()
        self.pi = nn.Linear(obs_dim, num_actions)
        self.v = nn.Linear(obs_dim, 1)

    def step(self, obs, state, action_mask=None):
        x = obs.reshape(obs.shape[0], -1)
        dist = CategoricalDist(self.pi(x))
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist, self.v(x).squeeze(-1), None)


def make_simple_network(obs_dim=8, hidden_dim=16, num_actions=4):
    """Legacy feedforward ActorCriticNetwork (deleted together with actor_critic.py)."""
    from colosseum.networks.actor_critic import ActorCriticNetwork

    encoder = SimpleEncoder(obs_dim, hidden_dim)
    policy = SimplePolicy(hidden_dim, num_actions)
    value = SimpleValue(hidden_dim)
    return ActorCriticNetwork(encoder, policy, value)


CORE_KINDS = ("none", "lstm", "gru", "window")


def make_core(kind: str, input_dim: int) -> Core:
    """Small core of each kind; recurrent sizes differ from input_dim on purpose (R1-21)."""
    if kind == "none":
        return NoCore(input_dim)
    if kind == "lstm":
        return LSTMCore(input_dim, hidden_size=24)
    if kind == "gru":
        return GRUCore(input_dim, hidden_size=24)
    if kind == "window":
        return WindowAttentionCore(input_dim, d_model=16, window=3, num_heads=2)
    raise ValueError(f"unknown core kind {kind!r}")


def make_simple_model(obs_dim: int = 8, hidden_dim: int = 16, num_actions: int = 4,
                      core: str = "none") -> ComposedModel:
    """ComposedModel: SimpleEncoder -> core -> SimplePolicy / SimpleValue."""
    encoder = SimpleEncoder(obs_dim, hidden_dim)
    trunk = make_core(core, encoder.latent_dim)
    return ComposedModel(encoder, trunk, SimplePolicy(trunk.output_dim, num_actions),
                         SimpleValue(trunk.output_dim))


def make_ttt_model() -> ComposedModel:
    """Tic-tac-toe example networks as a stateless ComposedModel."""
    from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue

    encoder = TicTacToeEncoder()
    return ComposedModel(encoder, NoCore(encoder.latent_dim), TicTacToePolicy(), TicTacToeValue())


# ---------------------------------------------------------------------------
# Deterministic toy environment for contract tests
# ---------------------------------------------------------------------------


class CountingEnv(BaseEnv):
    """Deterministic N-player simultaneous-move env for contract tests.

    Every episode lasts exactly ``episode_length`` steps and then terminates.
    At in-episode step ``t`` player ``p`` observes
    ``[t / episode_length, p, 1.0, 0.0, ...]`` (``obs_dim`` floats) and gets
    reward 1.0 if its action equals ``(t + p) % num_actions``, else 0.0.
    The step and player index can be recovered from any recorded observation:
    ``t = round(obs[0] * episode_length)``, ``p = round(obs[1])``.
    """

    def __init__(self, num_players: int = 2, episode_length: int = 5,
                 num_actions: int = 3, obs_dim: int = 4) -> None:
        if obs_dim < 3:
            raise ValueError("obs_dim must be >= 3")
        self._num_players = num_players
        self.episode_length = episode_length
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self._t = 0

    @property
    def num_players(self) -> int:
        return self._num_players

    @property
    def observation_space(self) -> gymnasium.spaces.Box:
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        return gymnasium.spaces.Discrete(self.num_actions)

    def _obs(self, p: int) -> np.ndarray:
        o = np.zeros(self.obs_dim, dtype=np.float32)
        o[0] = self._t / self.episode_length
        o[1] = float(p)
        o[2] = 1.0
        return o

    def reset(self, seed=None):
        self._t = 0
        players = range(self._num_players)
        return {p: self._obs(p) for p in players}, {p: {} for p in players}

    def step(self, actions):
        players = range(self._num_players)
        rewards = {p: 1.0 if int(actions[p]) == (self._t + p) % self.num_actions else 0.0 for p in players}
        self._t += 1
        done = self._t >= self.episode_length
        obs = {p: self._obs(p) for p in players}
        return obs, rewards, {p: done for p in players}, {p: False for p in players}, {p: {} for p in players}
```

- [ ] **Step 3: Run the new tests and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_registry.py -v`

Expected: collection error `ModuleNotFoundError: No module named 'colosseum.core.errors'` (and, for every module that imports `helpers`, `No module named 'colosseum.networks.composed'` — the rest of the suite is red until Step 7).

- [ ] **Step 4: Add the errors module**

Create `src/colosseum/core/errors.py`:

```python
"""Exception types raised by Colosseum."""

from __future__ import annotations


class ColosseumError(Exception):
    """Base class for all Colosseum errors."""


class ConfigError(ColosseumError):
    """The configuration is invalid or inconsistent (raised before any process starts)."""


class EnvContractError(ColosseumError):
    """An environment violated the ``BaseEnv`` contract (shapes, masks, flags)."""
```

- [ ] **Step 5: Add `ComposedModel`**

Create `src/colosseum/networks/composed.py`:

```python
"""ComposedModel: the default PolicyModel built from encoder, core and heads."""

from __future__ import annotations

from torch import Tensor

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.cores import Core
from colosseum.networks.model import PolicyModel, StepOutput, UnrollOutput
from colosseum.networks.state import State


class ComposedModel(PolicyModel):
    """``encoder(obs) -> core -> policy head / value head``.

    Submodules are named ``encoder``, ``core``, ``policy`` and ``value``
    (``NoCore`` has no parameters, so a stateless model's ``state_dict`` only
    holds ``encoder.*``, ``policy.*`` and ``value.*`` keys).
    """

    def __init__(self, encoder: BaseEncoder, core: Core, policy_head: BasePolicy, value_head: BaseValue) -> None:
        super().__init__()
        self.encoder = encoder
        self.core = core
        self.policy = policy_head
        self.value = value_head

    def initial_state(self, batch_size: int, device="cpu") -> State:
        return self.core.initial_state(batch_size, device)

    def step(self, obs: Tensor, state: State, action_mask: Tensor | None = None) -> StepOutput:
        features, next_state = self.core.step(self.encoder(obs), state)
        dist = self.policy(features)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        return StepOutput(dist=dist, value=self.value(features), state=next_state)

    def unroll(self, obs: Tensor, state0: State, dones: Tensor,
               action_mask: Tensor | None = None) -> UnrollOutput:
        T, B = obs.shape[0], obs.shape[1]
        latent = self.encoder(obs.reshape(T * B, *obs.shape[2:])).reshape(T, B, -1)
        features = self.core.unroll(latent, state0, dones).reshape(T * B, -1)
        dist = self.policy(features)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask.reshape(T * B, *action_mask.shape[2:]))
        return UnrollOutput(dist=dist, value=self.value(features))

    def reset_state(self, state: State, done: Tensor) -> State:
        return self.core.reset_state(state, done)
```

- [ ] **Step 6: Replace the `networks` config schema**

In `src/colosseum/core/config.py` replace the import line `from pydantic import BaseModel, Field` with `from pydantic import BaseModel, ConfigDict, Field, model_validator`, then replace the whole `NetworkConfig` class:

```python
class NetworkConfig(BaseModel):
    """Neural network architecture specification."""

    encoder_class: str = Field(..., description="Dotted path to the encoder class.")
    policy_class: str = Field(..., description="Dotted path to the policy head class.")
    value_class: str = Field(..., description="Dotted path to the value head class.")
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to network constructors.",
    )
    recurrent_type: str | None = Field(
        default=None,
        description="Recurrent trunk type: 'lstm', 'gru', or None (feedforward).",
    )
    recurrent_hidden_size: int = Field(default=128, ge=1, description="Hidden size for recurrent trunk.")
    recurrent_num_layers: int = Field(default=1, ge=1, description="Number of recurrent layers.")
```

with:

```python
class CoreConfig(BaseModel):
    """Core (trunk) between encoder and heads: ``{class: <dotted path>, kwargs: {...}}``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(
        ..., alias="class",
        description="Dotted path to a colosseum.networks.cores.Core subclass "
                    "(e.g. 'colosseum.networks.cores.LSTMCore').",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the core constructor (input_dim is passed automatically).",
    )


class NetworkConfig(BaseModel):
    """Model specification: a monolithic ``model_class`` or encoder + core + heads."""

    model_config = ConfigDict(extra="forbid")

    model_class: str | None = Field(
        default=None,
        description="Dotted path to a PolicyModel subclass. When set, encoder/core/heads must be omitted.",
    )
    encoder_class: str | None = Field(default=None, description="Dotted path to the encoder class.")
    core: CoreConfig | None = Field(
        default=None, description="Optional core between encoder and heads (null = stateless NoCore).",
    )
    policy_class: str | None = Field(default=None, description="Dotted path to the policy head class.")
    value_class: str | None = Field(default=None, description="Dotted path to the value head class.")
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to the model (or encoder and head) constructors.",
    )

    @model_validator(mode="after")
    def _check_model_spec(self) -> NetworkConfig:
        parts = {
            "encoder_class": self.encoder_class,
            "core": self.core,
            "policy_class": self.policy_class,
            "value_class": self.value_class,
        }
        if self.model_class:
            extra = [name for name, value in parts.items() if value is not None]
            if extra:
                raise ValueError(
                    f"networks.model_class is set, so {', '.join(extra)} must be omitted"
                )
            return self
        missing = [n for n in ("encoder_class", "policy_class", "value_class") if not parts[n]]
        if missing:
            raise ValueError(
                "networks: set either model_class, or all of encoder_class, policy_class, "
                f"value_class (missing: {', '.join(missing)})"
            )
        return self
```

`AgentConfig.networks` keeps the type `NetworkConfig | None`; `get_agent_config` round-trips `core` through `model_dump()`/`model_validate` because of `populate_by_name=True`.

- [ ] **Step 7: Rewrite `src/colosseum/core/registry.py`**

Replace the whole file with:

```python
"""Dynamic class import and instantiation utilities.

The registry module provides a thin convenience layer so that configuration
files can reference Python classes by their fully-qualified dotted path
(e.g. ``"examples.tic_tac_toe.env.TicTacToeEnv"``) and the framework can
import and instantiate them at runtime without hard-coded imports.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from colosseum.core.errors import ConfigError

if TYPE_CHECKING:
    from colosseum.core.config import ColosseumConfig
    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.model import PolicyModel


def import_class(dotted_path: str) -> type:
    """Import and return a class from a fully-qualified dotted path.

    Args:
        dotted_path: A string of the form ``"package.module.ClassName"``.

    Returns:
        The class object.

    Raises:
        ValueError: If *dotted_path* does not contain at least one ``'.'``
            separating a module path from a class name.
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the module does not contain the requested name.
    """
    if "." not in dotted_path:
        raise ValueError(
            f"dotted_path must be in the form 'module.ClassName', got: {dotted_path!r}"
        )

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(
            f"{dotted_path!r} resolved to {type(cls).__name__}, not a class"
        )

    return cls


def instantiate(dotted_path: str, **kwargs: Any) -> Any:
    """Import a class from *dotted_path* and return a new instance.

    This is a convenience wrapper around :func:`import_class` that also calls
    the constructor with the provided keyword arguments.

    Args:
        dotted_path: Fully-qualified class path.
        **kwargs: Arguments forwarded to the class constructor.

    Returns:
        An instance of the imported class.
    """
    cls = import_class(dotted_path)
    return cls(**kwargs)


def _accepts_in_dim(cls: type) -> bool:
    """True if ``cls.__init__`` declares an explicit ``in_dim`` parameter."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "in_dim" in params


def _build_head(dotted_path: str, in_dim: int, kwargs: dict[str, Any]) -> Any:
    """Instantiate a head, passing ``in_dim`` when its constructor accepts it."""
    cls = import_class(dotted_path)
    head_kwargs = dict(kwargs)
    if _accepts_in_dim(cls):
        head_kwargs["in_dim"] = in_dim
    return cls(**head_kwargs)


def build_model(config: ColosseumConfig) -> PolicyModel:
    """Build the agent's :class:`PolicyModel` from ``config.networks``.

    - ``model_class`` set: ``import_class(model_class)(**networks.kwargs)``; it must
      be a ``PolicyModel``.
    - otherwise a :class:`ComposedModel`: ``encoder(**kwargs)``, then
      ``core(input_dim=encoder.latent_dim, **core.kwargs)`` (``NoCore`` when
      ``core`` is null), then the heads with ``in_dim=core.output_dim`` if their
      constructor accepts ``in_dim``, plus ``**kwargs``.
    """
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.cores import Core, NoCore
    from colosseum.networks.model import PolicyModel

    net = config.networks
    if net.model_class:
        model = import_class(net.model_class)(**net.kwargs)
        if not isinstance(model, PolicyModel):
            raise ConfigError(
                f"networks.model_class {net.model_class!r} must subclass "
                f"colosseum.networks.model.PolicyModel, got {type(model).__name__}"
            )
        return model

    encoder = import_class(net.encoder_class)(**net.kwargs)
    latent_dim = encoder.latent_dim
    if net.core is None:
        core = NoCore(latent_dim)
    else:
        core = import_class(net.core.class_path)(input_dim=latent_dim, **net.core.kwargs)
        if not isinstance(core, Core):
            raise ConfigError(
                f"networks.core.class {net.core.class_path!r} must subclass "
                f"colosseum.networks.cores.Core, got {type(core).__name__}"
            )
    policy = _build_head(net.policy_class, core.output_dim, net.kwargs)
    value = _build_head(net.value_class, core.output_dim, net.kwargs)
    return ComposedModel(encoder, core, policy, value)


def build_network(config: ColosseumConfig) -> ActorCriticNetwork:
    """TRANSITIONAL: the legacy ``ActorCriticNetwork`` built from the new ``networks`` schema.

    Kept only until every caller uses :func:`build_model`; deleted together with
    ``networks/actor_critic.py``. Supports ``core`` = null, ``NoCore``,
    ``LSTMCore`` and ``GRUCore``.
    """
    import torch.nn as nn

    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.cores import GRUCore, LSTMCore, NoCore

    net = config.networks
    if net.model_class:
        raise ConfigError("networks.model_class is not supported by the legacy ActorCriticNetwork path")
    encoder = import_class(net.encoder_class)(**net.kwargs)
    out_dim = encoder.latent_dim
    recurrent = None
    if net.core is not None:
        core_cls = import_class(net.core.class_path)
        hidden = int(net.core.kwargs.get("hidden_size", 128))
        layers = int(net.core.kwargs.get("num_layers", 1))
        if issubclass(core_cls, LSTMCore):
            recurrent = nn.LSTM(out_dim, hidden, layers)
            out_dim = hidden
        elif issubclass(core_cls, GRUCore):
            recurrent = nn.GRU(out_dim, hidden, layers)
            out_dim = hidden
        elif not issubclass(core_cls, NoCore):
            raise ConfigError(f"core {net.core.class_path!r} is not supported by the legacy ActorCriticNetwork path")
    policy = _build_head(net.policy_class, out_dim, net.kwargs)
    value = _build_head(net.value_class, out_dim, net.kwargs)
    return ActorCriticNetwork(encoder, policy, value, recurrent=recurrent)


def _check_state(state: Any, batch: int, where: str) -> None:
    from colosseum.networks.state import tree_leaves

    for leaf in tree_leaves(state):
        if leaf.dim() == 0 or leaf.shape[0] != batch:
            raise ConfigError(
                f"{where}: every state tensor must have the batch dimension first "
                f"(expected {batch}), got a leaf of shape {tuple(leaf.shape)}"
            )


def validate_config(config: ColosseumConfig) -> None:
    """Build the env and the model and exercise ``step``/``unroll`` on dummy data.

    Raises :class:`ConfigError` with a precise message on any structural problem
    (bad class, head/core dimension mismatch, wrong value shape, state without a
    batch dim, action/mask size mismatch) before any process is spawned.
    """
    import numpy as np
    import torch

    from colosseum.core.action_spec import ActionSpec
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.distributions import Distribution

    try:
        env = import_class(config.env.env_class)(**config.env.kwargs)
    except Exception as e:
        raise ConfigError(f"Failed to create env {config.env.env_class!r}: {type(e).__name__}: {e}") from e
    try:
        try:
            model = build_model(config)
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"Failed to build model from config.networks: {type(e).__name__}: {e}") from e

        hint = ""
        if isinstance(model, ComposedModel):
            hint = (
                f" Policy/value heads receive core.output_dim={model.core.output_dim} features "
                f"(encoder.latent_dim={model.encoder.latent_dim}); give the heads an `in_dim` "
                f"constructor argument or size them to match."
            )

        spec = ActionSpec.from_space(env.action_space)
        obs_dict, info_dict = env.reset(seed=0)
        sample = torch.as_tensor(np.asarray(obs_dict[0], dtype=np.float32))
        B, T = 2, 3
        obs = sample.unsqueeze(0).repeat(B, *([1] * sample.dim()))
        mask = None
        raw_mask = info_dict.get(0, {}).get("action_mask") if isinstance(info_dict, dict) else None
        if raw_mask is not None:
            if isinstance(raw_mask, dict):
                flat_mask = spec.flatten_mask(raw_mask)
            else:
                flat_mask = np.asarray(raw_mask, dtype=bool)
            if flat_mask.shape != (spec.flat_mask_size,):
                raise ConfigError(
                    f"env action_mask has shape {flat_mask.shape}, but the action space needs "
                    f"({spec.flat_mask_size},)"
                )
            mask = torch.as_tensor(flat_mask).unsqueeze(0).repeat(B, 1)

        model.eval()
        with torch.no_grad():
            state0 = model.initial_state(B)
            _check_state(state0, B, "initial_state(2)")
            try:
                out = model.step(obs, state0, mask)
            except Exception as e:
                raise ConfigError(
                    f"model.step failed on a dummy batch with obs shape {tuple(obs.shape)}: "
                    f"{type(e).__name__}: {e}.{hint}"
                ) from e
            if not isinstance(out.dist, Distribution):
                raise ConfigError(
                    f"the policy head must return a colosseum Distribution "
                    f"(colosseum.networks.distributions), got {type(out.dist).__name__}"
                )
            if tuple(out.value.shape) != (B,):
                raise ConfigError(
                    f"value must have shape [B]=({B},), got {tuple(out.value.shape)}. "
                    f"Squeeze the last dim in the value head."
                )
            _check_state(out.state, B, "step() state")
            actions = out.dist.sample()
            expected = (B, *spec.action_shape)
            if tuple(actions.shape) != expected:
                raise ConfigError(
                    f"policy produced actions of shape {tuple(actions.shape)}, but the action space "
                    f"expects {expected}. Check the policy head / distribution."
                )

            dones = torch.zeros(T, B, dtype=torch.bool)
            dones[1, 0] = True
            obs_seq = obs.unsqueeze(0).repeat(T, *([1] * obs.dim()))
            mask_seq = None if mask is None else mask.unsqueeze(0).repeat(T, 1, 1)
            try:
                unrolled = model.unroll(obs_seq, state0, dones, mask_seq)
                log_probs = unrolled.dist.log_prob(actions.repeat(T, *([1] * (actions.dim() - 1))))
            except Exception as e:
                raise ConfigError(
                    f"model.unroll failed on a dummy [T={T}, B={B}] batch: {type(e).__name__}: {e}"
                ) from e
            if tuple(unrolled.value.shape) != (T * B,) or tuple(log_probs.shape) != (T * B,):
                raise ConfigError(
                    f"unroll must return time-major [T*B]=({T * B},) values/log-probs, got "
                    f"{tuple(unrolled.value.shape)} / {tuple(log_probs.shape)}"
                )
    finally:
        env.close()
```

- [ ] **Step 8: Let the example heads take `in_dim`**

1. In `examples/tic_tac_toe/networks.py` replace:

```python
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Linear(64, 9)
```

   with:

```python
    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, 9)
```

2. In `examples/tic_tac_toe/networks.py` replace:

```python
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
```

   with:

```python
    def __init__(self, in_dim: int = 64, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
```

3. In `examples/composite_action/networks.py` replace:

```python
    def __init__(self):
        super().__init__()
        self.dir_head = nn.Linear(_LATENT, 4)
        self.speed_mean = nn.Linear(_LATENT, 1)
```

   with:

```python
    def __init__(self, in_dim: int = _LATENT):
        super().__init__()
        self.dir_head = nn.Linear(in_dim, 4)
        self.speed_mean = nn.Linear(in_dim, 1)
```

4. In `examples/composite_action/networks.py` replace:

```python
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_LATENT, 16),
```

   with:

```python
    def __init__(self, in_dim: int = _LATENT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
```

5. In `examples/space_miners/networks.py` replace:

```python
    def __init__(self, **kwargs) -> None:
        super().__init__()
        # Acceleration: 3 ships × 2D = 6
        self.accel_mean = nn.Linear(_LATENT, 6)
        self.accel_logstd = nn.Parameter(torch.zeros(6))

        # Push: binary per ship
        self.push_head_0 = nn.Linear(_LATENT, 2)
        self.push_head_1 = nn.Linear(_LATENT, 2)
        self.push_head_2 = nn.Linear(_LATENT, 2)
```

   with:

```python
    def __init__(self, in_dim: int = _LATENT, **kwargs) -> None:
        super().__init__()
        # Acceleration: 3 ships × 2D = 6
        self.accel_mean = nn.Linear(in_dim, 6)
        self.accel_logstd = nn.Parameter(torch.zeros(6))

        # Push: binary per ship
        self.push_head_0 = nn.Linear(in_dim, 2)
        self.push_head_1 = nn.Linear(in_dim, 2)
        self.push_head_2 = nn.Linear(in_dim, 2)
```

6. In `examples/space_miners/networks.py` replace:

```python
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_LATENT, 64),
```

   with:

```python
    def __init__(self, in_dim: int = _LATENT, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
```

- [ ] **Step 9: Add `core: null` to every example config**

```bash
sed -i 's/^\(  encoder_class: .*\)$/\1\n  core: null              # e.g. {class: colosseum.networks.cores.LSTMCore, kwargs: {hidden_size: 128}}/' configs/examples/*.yaml
```

Check: `grep -n "core:" configs/examples/*.yaml` shows one line per file, right below `encoder_class`. No example used the removed keys: `grep -rn "recurrent_" configs examples` prints nothing.

- [ ] **Step 10: Migrate the existing tests that used the old schema or head signature**

In `tests/unit/test_recurrent.py` replace the three tests `test_build_network_with_lstm`, `test_build_network_with_gru`, `test_build_network_feedforward_default`:

```python
def test_build_network_with_lstm():
    """build_network should create LSTM trunk when recurrent_type='lstm'."""
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
    from colosseum.core.registry import build_network

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
        recurrent_type="lstm",
        recurrent_hidden_size=HIDDEN_SIZE,
        recurrent_num_layers=NUM_LAYERS,
    )
    env_cfg = EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv")
    config = ColosseumConfig(env=env_cfg, networks=net_cfg)

    net = build_network(config)
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.LSTM)
    assert net.recurrent_hidden_size == HIDDEN_SIZE
    assert net.recurrent_num_layers == NUM_LAYERS


def test_build_network_with_gru():
    """build_network should create GRU trunk when recurrent_type='gru'."""
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
    from colosseum.core.registry import build_network

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
        recurrent_type="gru",
        recurrent_hidden_size=HIDDEN_SIZE,
        recurrent_num_layers=1,
    )
    env_cfg = EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv")
    config = ColosseumConfig(env=env_cfg, networks=net_cfg)

    net = build_network(config)
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.GRU)


def test_build_network_feedforward_default():
    """build_network with no recurrent_type should create feedforward net."""
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig
    from colosseum.core.registry import build_network

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
    )
    env_cfg = EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv")
    config = ColosseumConfig(env=env_cfg, networks=net_cfg)

    net = build_network(config)
    assert not net.is_recurrent
    assert net.recurrent is None
```

with:

```python
def _legacy_cfg(core=None):
    from colosseum.core.config import ColosseumConfig, EnvConfig, NetworkConfig

    net_cfg = NetworkConfig(
        encoder_class="helpers.SimpleEncoder",
        core=core,
        policy_class="helpers.SimplePolicy",
        value_class="helpers.SimpleValue",
    )
    return ColosseumConfig(env=EnvConfig(env_class="examples.tic_tac_toe.env.TicTacToeEnv"), networks=net_cfg)


def test_build_network_with_lstm():
    """Transitional build_network maps an LSTMCore config to an nn.LSTM trunk."""
    from colosseum.core.registry import build_network

    net = build_network(_legacy_cfg({"class": "colosseum.networks.cores.LSTMCore",
                                     "kwargs": {"hidden_size": 24, "num_layers": NUM_LAYERS}}))
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.LSTM)
    assert net.recurrent_hidden_size == 24
    assert net.recurrent_num_layers == NUM_LAYERS
    assert net.policy.fc.in_features == 24  # in_dim passed to the head


def test_build_network_with_gru():
    """Transitional build_network maps a GRUCore config to an nn.GRU trunk."""
    from colosseum.core.registry import build_network

    net = build_network(_legacy_cfg({"class": "colosseum.networks.cores.GRUCore",
                                     "kwargs": {"hidden_size": 24}}))
    assert net.is_recurrent
    assert isinstance(net.recurrent, nn.GRU)


def test_build_network_feedforward_default():
    """Transitional build_network without a core builds a feedforward net."""
    from colosseum.core.registry import build_network

    net = build_network(_legacy_cfg())
    assert not net.is_recurrent
    assert net.recurrent is None
```

In `tests/unit/test_review_fixes.py` (inside `_make_recurrent_net`) replace `pol = SimplePolicy(hidden_dim=hidden, num_actions=num_actions)` with `pol = SimplePolicy(in_dim=hidden, num_actions=num_actions)` and `val = SimpleValue(hidden_dim=hidden)` with `val = SimpleValue(in_dim=hidden)`.

- [ ] **Step 11: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_registry.py tests/unit/test_recurrent.py tests/unit/test_review_fixes.py -v`

Expected: all pass (`test_registry.py`: 17 tests).

- [ ] **Step 12: Validate every example config from the CLI**

Run: `for c in tic_tac_toe tic_tac_toe_multi chase; do PYTHONPATH=. .venv/bin/colosseum validate -c configs/examples/$c.yaml; done`

Expected: `Config is valid.` three times (`PYTHONPATH=.` because the CLI does not put the working directory on `sys.path` before T6.5; `space_miners.yaml` additionally needs Box2D from the `examples` extra).

- [ ] **Step 13: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!`; all tests pass. Training still uses `ActorCriticNetwork` (via the transitional `build_network`) in this task.

- [ ] **Step 14: Commit**

```bash
git add src/colosseum/core src/colosseum/networks/composed.py examples configs tests
git commit -m "feat: ComposedModel, networks schema with cores, build_model and strict validate_config"
```

---

### Task T1.5: APPO on `PolicyModel.unroll`; `TrajectoryChunk.initial_state`; worker and launcher on `PolicyModel`

Findings: R1-01 / R2-01 (recurrent training crashed on worker-built chunks: hidden layout `[L,1,H]` vs `[L,B,H]`), R1-02 (RNN silently bypassed), ET-10.

**Scope note (see "Contract notes").** Removing `TrajectoryChunk.lstm_hidden` and switching APPO to `PolicyModel` cannot be done without switching the producer of chunks and weights in the same commit: the worker writes `lstm_hidden` and loads the learner's `state_dict`. So this task switches the whole training path (learner, worker loop, launcher, distributed mode) to `PolicyModel`. T1.6 then adds the four-core worker → learner contract test on top. Eval, BC and kickstart keep their current code paths until T1.7 (BC/eval CLI still build the transitional `ActorCriticNetwork`; kickstart and BC already work with `ComposedModel` because its submodules are named `encoder`/`policy`).

**Files:**
- Modify: `src/colosseum/core/types.py` (`TrajectoryChunk`), `src/colosseum/transport/serialization.py`, `src/colosseum/algorithms/base.py`, `src/colosseum/algorithms/appo.py` (rewrite), `src/colosseum/learner/learner.py` (lines 65, 125, 200), `src/colosseum/worker/rollout_loop.py`, `src/colosseum/worker/rollout_worker.py`, `src/colosseum/launcher.py`, `src/colosseum/distributed.py`
- Modify (tests): `tests/contract/harness.py`, `tests/unit/test_appo.py`, `tests/unit/test_bc.py`, `tests/unit/test_composite_actions.py`, `tests/unit/test_action_masking.py`, `tests/unit/test_performance.py`, `tests/unit/test_review_fixes.py`, `tests/unit/test_recurrent.py`, `tests/integration/test_pipelines.py`
- Test (new): `tests/unit/test_appo_unroll.py`, `tests/contract/test_rollout_state.py`

**Interfaces:**
- Consumes: T1.1 (`cat_batch`, `slice_batch`, `state_to`, `tree_map`, `State`), T1.2 (`PolicyModel`, `act`), T1.4 (`build_model`, `ComposedModel`, helpers `make_simple_model`, `make_ttt_model`, `CORE_KINDS`), T0.5 (`RolloutLoop`, harness).
- Produces:
  - Contract `TrajectoryChunk`: field `initial_state: State = None` (leaves `[1, ...]`) replaces `lstm_hidden`; `to(device)` replaces `to_device(device)` and moves state leaves too; `pin_memory()` unchanged. (`to_payload`/`from_payload` come in T2.2.)
  - Contract `BaseAlgorithm.model -> PolicyModel` (abstract property; replaces `network`).
  - Contract `APPO.__init__(self, model: PolicyModel, config: AlgorithmConfig, device="cpu", pin_memory=False, kickstart: KickstartLoss | None = None)`; property `model`. Training always runs `model.unroll(obs [T,B,...], cat_batch(initial_states), dones.bool(), action_masks)`; there is no separate recurrent path. `setup_lr_schedule` stays until T2.5.
  - New: `APPO.evaluate_chunks(chunks: list[TrajectoryChunk]) -> tuple[Tensor, Tensor]` — current model's log-probs of the recorded actions and values, each `[T*B]` time-major (index `t*B + b` = step `t` of `chunks[b]`), computed exactly as in training, under `torch.no_grad()`.
  - `RolloutBuffer.initial_state: State` and `RolloutBuffer.set_initial_state(state)` replace `set_lstm_init`/`_lstm_*_init`.
  - `RolloutLoop(..., model_factories: dict[str, Callable[[], PolicyModel]], ...)`: keeps a `State` per (env, slot) in `_slot_states` (leaves `[1, ...]`), grouped inference does `cat_batch` → `act` → `slice_batch`; at chunk start the slot state is stored as `initial_state`; at episode end every slot of the env gets `initial_state(1)` of the (possibly re-assigned) agent's model; the chunk-end bootstrap uses `act(model, next_obs, slot_state)`.
  - `_run_inference_group(model, indices, obs_flat, all_masks, slot_states, out_actions, out_log_probs, out_values)`, `_apply_command(cmd, models_by_agent, model_factories, pending)`.
  - `rollout_worker_process(..., model_factories: dict[str, Callable[[], PolicyModel]], ...)` (keyword renamed from `network_factories`).
  - `colosseum.launcher._create_model(config) -> PolicyModel` (replaces `_create_network`); launcher and distributed learners/workers build models with `build_model`.
  - Learner uses `algorithm.model` (load/push/checkpoint).
  - gRPC chunk serialization carries `initial_state` (any dict/tuple/list of tensors).
  - `tests/contract/harness.py`: `simple_factory(core: str = "none")` returns `make_simple_model(obs_dim=4, hidden_dim=16, num_actions=3, core=core)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_appo_unroll.py`:

```python
"""APPO trains every PolicyModel through model.unroll from the chunks' initial states."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import TrajectoryChunk
from colosseum.networks.state import slice_batch, tree_leaves
from colosseum.transport.serialization import deserialize_chunk, serialize_chunk
from helpers import CORE_KINDS, make_simple_model

OBS, A, T = 8, 4, 6


def _chunk(model, initial_state, dones=None, masks=None) -> TrajectoryChunk:
    d = torch.zeros(T) if dones is None else dones
    return TrajectoryChunk(
        agent_id="a",
        observations=torch.randn(T, OBS),
        actions=torch.randint(0, A, (T,)),
        action_log_probs=torch.full((T,), -1.3),
        rewards=torch.randn(T),
        dones=d,
        values=torch.randn(T),
        bootstrap_value=torch.tensor(0.0),
        behavior_policy_version=0,
        initial_state=initial_state,
        action_masks=masks,
    )


def _reference(model, chunks):
    """Per-chunk step loop with resets: expected [T*B] time-major log-probs/values."""
    lps = torch.zeros(T, len(chunks))
    vals = torch.zeros(T, len(chunks))
    with torch.no_grad():
        for b, c in enumerate(chunks):
            state = c.initial_state
            for t in range(T):
                mask = None if c.action_masks is None else c.action_masks[t:t + 1]
                out = model.step(c.observations[t:t + 1], state, mask)
                lps[t, b] = out.dist.log_prob(c.actions[t:t + 1])[0]
                vals[t, b] = out.value[0]
                state = out.state
                if bool(c.dones[t]):
                    state = model.initial_state(1)
    return lps.reshape(-1), vals.reshape(-1)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_evaluate_chunks_matches_step_loop(core):
    torch.manual_seed(0)
    model = make_simple_model(obs_dim=OBS, num_actions=A, core=core)
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    warm = model.initial_state(2)
    if warm is not None:
        with torch.no_grad():
            for _ in range(3):
                warm = model.step(torch.randn(2, OBS), warm).state
    dones = torch.zeros(T)
    dones[2] = 1.0
    masks = torch.ones(T, A, dtype=torch.bool)
    masks[:, 3] = False
    chunks = [
        _chunk(model, None if warm is None else slice_batch(warm, 0), dones=dones),
        _chunk(model, None if warm is None else slice_batch(warm, 1), masks=masks),
    ]
    chunks[0].action_masks = torch.ones(T, A, dtype=torch.bool)
    chunks[1].actions = torch.randint(0, 3, (T,))
    lp, v = algo.evaluate_chunks(chunks)
    ref_lp, ref_v = _reference(model, chunks)
    assert lp.shape == (T * 2,) and v.shape == (T * 2,)
    assert torch.allclose(lp, ref_lp, atol=1e-5)
    assert torch.allclose(v, ref_v, atol=1e-5)


@pytest.mark.parametrize("core", CORE_KINDS)
def test_train_step_on_stateful_chunks_updates_all_parameters(core):
    torch.manual_seed(0)
    model = make_simple_model(obs_dim=OBS, num_actions=A, core=core)
    algo = APPO(model, AlgorithmConfig(learning_rate=1e-2), device="cpu")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    chunks = [_chunk(model, model.initial_state(1)) for _ in range(3)]
    metrics = algo.train_step(chunks)
    assert np.isfinite(metrics["total_loss"])
    assert algo.policy_version == 1
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert {k for k, _ in model.named_parameters()} <= changed


def test_algorithm_exposes_model():
    model = make_simple_model()
    assert APPO(model, AlgorithmConfig()).model is model


def test_chunk_to_and_serialization_keep_initial_state():
    model = make_simple_model(obs_dim=OBS, num_actions=A, core="lstm")
    state = model.initial_state(1)
    state = {k: v + 0.5 for k, v in state.items()}
    chunk = _chunk(model, state)
    moved = chunk.to("cpu")
    assert torch.equal(moved.initial_state["h"], state["h"])
    data, compressed = serialize_chunk(chunk)
    restored = deserialize_chunk("a", 0, data, compressed)
    assert set(restored.initial_state) == {"h", "c"}
    for x, y in zip(tree_leaves(restored.initial_state), tree_leaves(state)):
        assert torch.equal(x, y)
    stateless = _chunk(model, None)
    data, compressed = serialize_chunk(stateless)
    assert deserialize_chunk("a", 0, data, compressed).initial_state is None
```

In `tests/contract/harness.py`:

1. Replace:

```python
from helpers import CountingEnv, make_simple_network
```

   with:

```python
from helpers import CountingEnv, make_simple_model
```

2. Replace:

```python
def simple_factory() -> Any:
    """Model factory used by the contract tests (small MLP actor-critic)."""
    return make_simple_network(obs_dim=OBS_DIM, hidden_dim=16, num_actions=NUM_ACTIONS)
```

   with:

```python
def simple_factory(core: str = "none") -> Any:
    """Model factory used by the contract tests: SimpleEncoder -> core -> heads."""
    return make_simple_model(obs_dim=OBS_DIM, hidden_dim=16, num_actions=NUM_ACTIONS, core=core)
```

Create `tests/contract/test_rollout_state.py`:

```python
"""RolloutLoop keeps an opaque per-slot State and records it at chunk start."""

from __future__ import annotations

import pytest
import torch

from harness import make_loop, run_steps, simple_factory
from helpers import CORE_KINDS


@pytest.mark.parametrize("core", CORE_KINDS)
def test_chunks_carry_initial_state_with_batch_one(core):
    torch.manual_seed(0)
    loop, rec = make_loop(model_factories={"agent_0": lambda: simple_factory(core)},
                          num_envs=2, chunk_length=4)
    run_steps(loop, 8)
    loop.close()
    assert len(rec.chunks) == 8
    for c in rec.chunks:
        if core == "none":
            assert c.initial_state is None
        else:
            leaves = list(c.initial_state.values())
            assert all(leaf.shape[0] == 1 for leaf in leaves)
    if core != "none":
        # The first chunk of each slot starts at an episode start (zero state);
        # the second one starts mid-episode (non-zero state).
        first, second = rec.chunks[:4], rec.chunks[4:]
        assert all(float(c.initial_state[k].abs().sum()) == 0.0 for c in first for k in ("h",) if k in c.initial_state)
        assert any(float(sum(v.float().abs().sum() for v in c.initial_state.values())) > 0 for c in second)


def test_slot_state_is_reset_at_episode_end():
    torch.manual_seed(0)
    loop, _ = make_loop(model_factories={"agent_0": lambda: simple_factory("gru")},
                        num_envs=1, chunk_length=4)
    run_steps(loop, 4)
    assert float(loop._slot_states[(0, 0)]["h"].abs().sum()) > 0
    run_steps(loop, 1)  # 5th step ends the episode (episode_length=5)
    assert float(loop._slot_states[(0, 0)]["h"].abs().sum()) == 0.0
    loop.close()
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_appo_unroll.py tests/contract/test_rollout_state.py -v`

Expected: failures such as `TypeError: TrajectoryChunk.__init__() got an unexpected keyword argument 'initial_state'`, `AttributeError: 'APPO' object has no attribute 'evaluate_chunks'` and, in the rollout tests, `AttributeError: 'ComposedModel' object has no attribute 'is_recurrent'`.

- [ ] **Step 3: Replace `lstm_hidden` with `initial_state` in `TrajectoryChunk`**

In `src/colosseum/core/types.py` add the import below `import torch` (keep the blank line after it):

```python
from colosseum.networks.state import State, tree_map
```

and replace the whole `TrajectoryChunk` class (from `@dataclass` / `class TrajectoryChunk:` down to, not including, `@dataclass` / `class PlayerSlot:`) with:

```python
@dataclass
class TrajectoryChunk:
    """Fixed-length rollout chunk from a single agent slot in a single env.

    Workers collect observations, actions, rewards, etc. over T consecutive
    timesteps and package them into chunks that are sent to learners for
    training.  Episode boundaries are handled within chunks: ``dones[t]``
    marks the end of an episode, and the next timestep starts a new one.

    Attributes:
        agent_id: Identifier of the agent that generated this chunk.
        observations: Stacked observations of shape ``[T, *obs_shape]``.
        actions: Actions taken at each step, shape ``[T, *act_shape]``.
        action_log_probs: Log-probabilities of the chosen actions under the
            behavior policy, shape ``[T]``.
        rewards: Scalar rewards received after each action, shape ``[T]``.
        dones: Episode-termination flags, shape ``[T]`` (transition t is the
            last of its episode).
        values: Value estimates from the behavior policy, shape ``[T]``.
        bootstrap_value: Value estimate after the last transition (scalar),
            0 when the last transition is terminal.
        behavior_policy_version: Version counter of the policy that was used
            to collect this chunk.
        initial_state: Model state (a ``State`` pytree, see
            ``colosseum.networks.state``) before the chunk's first transition;
            every tensor leaf has batch dim 1 (``[1, ...]``). ``None`` for
            stateless models. The learner concatenates these along dim 0 and
            unrolls the model from them.
        action_masks: Optional ``[T, mask_size]`` bool masks the behavior
            policy acted under.
    """

    agent_id: str
    observations: torch.Tensor  # [T, *obs_shape]
    actions: torch.Tensor  # [T, *act_shape]
    action_log_probs: torch.Tensor  # [T]
    rewards: torch.Tensor  # [T]
    dones: torch.Tensor  # [T] bool
    values: torch.Tensor  # [T]
    bootstrap_value: torch.Tensor  # scalar
    behavior_policy_version: int
    initial_state: State = None  # leaves [1, ...]; None for stateless models
    action_masks: torch.Tensor | None = None  # [T, mask_size]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def chunk_length(self) -> int:
        """Number of timesteps ``T`` in this chunk."""
        return self.observations.shape[0]

    def _apply_to_tensors(self, fn) -> TrajectoryChunk:
        """Return a copy with *fn* applied to every tensor (including state leaves)."""
        return TrajectoryChunk(
            agent_id=self.agent_id,
            observations=fn(self.observations),
            actions=fn(self.actions),
            action_log_probs=fn(self.action_log_probs),
            rewards=fn(self.rewards),
            dones=fn(self.dones),
            values=fn(self.values),
            bootstrap_value=fn(self.bootstrap_value),
            behavior_policy_version=self.behavior_policy_version,
            initial_state=tree_map(fn, self.initial_state),
            action_masks=fn(self.action_masks) if self.action_masks is not None else None,
        )

    def to(self, device: str | torch.device) -> TrajectoryChunk:
        """Return a copy with all tensors moved to *device*."""
        return self._apply_to_tensors(lambda t: t.to(device))

    def pin_memory(self) -> TrajectoryChunk:
        """Pin all tensors to page-locked memory for faster host-to-device copies."""
        return self._apply_to_tensors(lambda t: t.pin_memory())
```

- [ ] **Step 4: Carry the state through gRPC serialization**

In `src/colosseum/transport/serialization.py`:

1. Replace:

```python
    if chunk.lstm_hidden is not None:
        tensor_dict["lstm_h"] = chunk.lstm_hidden[0]
        tensor_dict["lstm_c"] = chunk.lstm_hidden[1]
```

   with:

```python
    if chunk.initial_state is not None:
        tensor_dict["initial_state"] = chunk.initial_state  # dict/tuple of tensors: weights_only-safe
```

2. Delete:

```python
    lstm_hidden = None
    if "lstm_h" in td and "lstm_c" in td:
        lstm_hidden = (td["lstm_h"], td["lstm_c"])
```

3. Replace:

```python
        lstm_hidden=lstm_hidden,
```

   with:

```python
        initial_state=td.get("initial_state"),
```

- [ ] **Step 5: `BaseAlgorithm.model`**

In `src/colosseum/algorithms/base.py`:

1. Replace:

```python
    from colosseum.networks.actor_critic import ActorCriticNetwork
```

   with:

```python
    from colosseum.networks.model import PolicyModel
```

2. Replace:

```python
    Subclasses must implement: compute_loss, train_step, network, policy_version.
```

   with:

```python
    Subclasses must implement: compute_loss, train_step, model, policy_version.
```

3. Replace:

```python
    def network(self) -> ActorCriticNetwork:
        """The neural network being trained."""
```

   with:

```python
    def model(self) -> PolicyModel:
        """The PolicyModel being trained."""
```

- [ ] **Step 6: Rewrite `src/colosseum/algorithms/appo.py`**

Replace the whole file with (changes: `model`/`PolicyModel` instead of `network`/`ActorCriticNetwork`, `kickstart` after `pin_memory`, `initial_state` batched in `_prepare_batch`, one `_evaluate` path through `model.unroll`, new `evaluate_chunks`):

```python
"""APPO: Async PPO with V-trace off-policy correction.

Combines:
- V-trace targets for value function learning (handles policy lag from async workers)
- PPO clipped surrogate for policy gradient (additional stability)
- Entropy bonus for exploration

This is the primary algorithm used by Sample Factory, PufferLib, and similar
to OpenAI Five's approach.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.algorithms.vtrace import compute_vtrace
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig, LRSchedule
from colosseum.core.types import TrajectoryChunk
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import cat_batch, state_to


class APPO(BaseAlgorithm):
    """Async PPO with V-trace off-policy correction."""

    def __init__(
        self,
        model: PolicyModel,
        config: AlgorithmConfig,
        device: str | torch.device = "cpu",
        pin_memory: bool = False,
        kickstart: KickstartLoss | None = None,
    ):
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._policy_version = 0
        self._kickstart = kickstart
        self._pin_memory = pin_memory

        # AMP (automatic mixed precision)
        self._use_amp = config.use_amp and str(device).startswith("cuda")
        self._amp_dtype = getattr(torch, config.amp_dtype, torch.float16)
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._zero_loss = torch.tensor(0.0, device=device)

        # Optionally compile V-trace for faster execution
        if config.use_torch_compile:
            self._compute_vtrace = torch.compile(compute_vtrace)
        else:
            self._compute_vtrace = compute_vtrace

        # LR scheduler (created in setup_lr_schedule when total_steps is known)
        self._lr_scheduler = None

    def setup_lr_schedule(self, total_steps: int) -> None:
        """Set up LR schedule over total training steps."""
        if self._config.lr_schedule == LRSchedule.LINEAR:
            self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / max(1, total_steps)),
            )
        elif self._config.lr_schedule == LRSchedule.COSINE:
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=max(1, total_steps),
            )
        # CONSTANT: no scheduler needed

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def optimizer_state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def _prepare_batch(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """Stack trajectory chunks into batched tensors.

        Each chunk has shape [T, ...]. We stack them to get [T, B, ...] where B = len(chunks).
        If pin_memory is enabled and device is CUDA, tensors are pinned for
        async DMA transfer.
        """
        device = self._device
        use_pinning = self._pin_memory and str(device).startswith("cuda")

        batch_cpu = {
            "observations": torch.stack([c.observations for c in chunks], dim=1),
            "actions": torch.stack([c.actions for c in chunks], dim=1),
            "behavior_log_probs": torch.stack([c.action_log_probs for c in chunks], dim=1),
            "rewards": torch.stack([c.rewards for c in chunks], dim=1),
            "dones": torch.stack([c.dones for c in chunks], dim=1),
            "old_values": torch.stack([c.values for c in chunks], dim=1),
            "bootstrap_values": torch.stack([c.bootstrap_value for c in chunks]),
        }
        if chunks[0].action_masks is not None:
            batch_cpu["action_masks"] = torch.stack(
                [c.action_masks for c in chunks], dim=1,
            )

        if use_pinning:
            batch = {
                k: v.pin_memory().to(device, non_blocking=True)
                for k, v in batch_cpu.items()
            }
        else:
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
        # Model state before each chunk's first transition: leaves [1, ...] -> [B, ...].
        batch["initial_state"] = state_to(cat_batch([c.initial_state for c in chunks]), device)
        return batch

    def _evaluate(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll the model over the batch: (log_probs, values, entropy), each ``[T, B]``."""
        T, B = batch["rewards"].shape
        out = self._model.unroll(
            batch["observations"],
            batch["initial_state"],
            batch["dones"].bool(),
            batch.get("action_masks"),
        )
        actions = batch["actions"].reshape(T * B, *batch["actions"].shape[2:])
        log_probs = out.dist.log_prob(actions).reshape(T, B)
        values = out.value.reshape(T, B)
        entropy = out.dist.entropy().reshape(T, B)
        return log_probs, values, entropy

    @torch.no_grad()
    def evaluate_chunks(self, chunks: list[TrajectoryChunk]) -> tuple[torch.Tensor, torch.Tensor]:
        """Current model's log-probs of the recorded actions and its values.

        Returns two ``[T*B]`` tensors in time-major order (index ``t*B + b`` is
        step ``t`` of ``chunks[b]``), computed exactly as in training:
        ``model.unroll`` from ``cat_batch(chunk.initial_state ...)``.
        """
        log_probs, values, _ = self._evaluate(self._prepare_batch(chunks))
        return log_probs.reshape(-1), values.reshape(-1)

    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """Compute APPO loss from a batch of trajectory chunks.

        Steps:
        1. Stack chunks into [T, B, ...] tensors
        2. Forward pass: ``model.unroll`` from the chunks' initial states gives
           current log_probs, values, entropy (one path for every model)
        3. Compute V-trace targets + advantages
        4. PPO clipped surrogate with V-trace advantages
        5. Value loss: MSE(values, vtrace_targets)
        6. Entropy bonus
        """
        cfg = self._config
        batch = self._prepare_batch(chunks)

        T, B = batch["rewards"].shape
        obs_shape = batch["observations"].shape[2:]
        # Flattened observations [T*B, *obs_shape] for the kickstart KL term.
        flat_obs = batch["observations"].reshape(T * B, *obs_shape)

        amp_ctx = torch.autocast(
            device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp,
        )
        with amp_ctx:
            target_log_probs, new_values, entropy = self._evaluate(batch)

        # V-trace targets and advantages
        with torch.no_grad():
            vtrace_targets, vtrace_advantages = self._compute_vtrace(
                behavior_log_probs=batch["behavior_log_probs"],
                target_log_probs=target_log_probs.detach(),
                rewards=batch["rewards"],
                values=new_values.detach(),
                bootstrap_value=batch["bootstrap_values"],
                dones=batch["dones"],
                gamma=cfg.gamma,
                rho_bar=cfg.vtrace_rho_bar,
                c_bar=cfg.vtrace_c_bar,
            )

        # PPO clipped surrogate loss
        log_ratio = torch.clamp(
            target_log_probs - batch["behavior_log_probs"], -20.0, 20.0
        )
        ratio = torch.exp(log_ratio)
        adv = vtrace_advantages.detach()
        if cfg.normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, vtrace_targets.detach())

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = policy_loss + cfg.value_loss_coeff * value_loss + cfg.entropy_coeff * entropy_loss

        # Kickstart loss (KL to teacher policy)
        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0:
            kickstart_loss = self._kickstart.compute(self._model, flat_obs)
            total_loss = total_loss + kickstart_loss

        # Metrics for logging
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.eps_clip).float().mean()

        result = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": -entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        if self._kickstart is not None:
            result["kickstart_loss"] = kickstart_loss.detach()
            result["kickstart_lambda"] = torch.tensor(self._kickstart.current_lambda)
        return result

    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]:
        """Full training step with minibatch iterations.

        Performs num_epochs passes over the data, splitting into minibatches.
        """
        cfg = self._config
        metrics_accum: dict[str, float] = {}
        num_updates = 0

        for _epoch in range(cfg.num_epochs):
            # For APPO, we typically do a single pass (num_epochs=1)
            # because the data is already off-policy. Multiple epochs
            # further increase the off-policyness.
            # However, we support multiple epochs for flexibility.

            # Shuffle chunks and create minibatches (minibatching is over the
            # batch dimension B = number of chunks; each chunk's [T] sequence
            # stays intact so recurrent training is unaffected).
            indices = torch.randperm(len(chunks))
            mb_size = cfg.minibatch_chunks if cfg.minibatch_chunks > 0 else len(chunks)

            for start in range(0, len(chunks), mb_size):
                end = min(start + mb_size, len(chunks))
                mb_indices = indices[start:end]
                mb_chunks = [chunks[i] for i in mb_indices]

                if not mb_chunks:
                    continue

                losses = self.compute_loss(mb_chunks)
                total_loss = losses["total_loss"]

                self._optimizer.zero_grad()

                if self._scaler is not None:
                    self._scaler.scale(total_loss).backward()
                    if cfg.max_grad_norm > 0:
                        self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), cfg.max_grad_norm)
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    total_loss.backward()
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), cfg.max_grad_norm)
                    self._optimizer.step()

                # Accumulate metrics
                for key, value in losses.items():
                    if key not in metrics_accum:
                        metrics_accum[key] = 0.0
                    metrics_accum[key] += value.item()
                num_updates += 1

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        if self._kickstart is not None:
            self._kickstart.step()

        self._policy_version += 1

        # Average metrics
        metrics = {k: v / max(1, num_updates) for k, v in metrics_accum.items()}
        metrics["policy_version"] = float(self._policy_version)
        metrics["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        return metrics
```

- [ ] **Step 7: Learner uses `algorithm.model`**

Run: `sed -i 's/algorithm\.network\./algorithm.model./g' src/colosseum/learner/learner.py`

Check: `grep -n "algorithm\.model\." src/colosseum/learner/learner.py` shows lines 65, 125 and 200; `grep -n "\.network" src/colosseum/learner/learner.py` prints nothing.

- [ ] **Step 8: Per-slot `State` in the rollout loop**

In `src/colosseum/worker/rollout_loop.py` make these replacements (they remove every `is_recurrent` / `initial_hidden` / `hidden_states` / `_any_recurrent` use):

1. Replace:

```python
import numpy as np
import torch
import torch.nn as nn

from colosseum.core.action_spec import ActionSpec
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
```

   with:

```python
import numpy as np
import torch

from colosseum.core.action_spec import ActionSpec
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.envs.base_env import BaseEnv
from colosseum.envs.vec_env import VectorEnv
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
```

2. Replace:

```python
    _lstm_h_init: torch.Tensor | None = field(init=False, default=None, repr=False)
    _lstm_c_init: torch.Tensor | None = field(init=False, default=None, repr=False)
```

   with:

```python
    initial_state: State = field(init=False, default=None, repr=False)
```

3. Replace:

```python
    def set_lstm_init(self, h: torch.Tensor, c: torch.Tensor):
        """Save initial LSTM hidden state for the current chunk."""
        self._lstm_h_init = h.clone()
        self._lstm_c_init = c.clone()

    def reset(self):
        self._cursor = 0
        self._has_masks = False
        self._lstm_h_init = None
        self._lstm_c_init = None
```

   with:

```python
    def set_initial_state(self, state: State) -> None:
        """Record the model state before the chunk's first transition (leaves [1, ...])."""
        self.initial_state = state

    def reset(self):
        self._cursor = 0
        self._has_masks = False
        self.initial_state = None
```

4. Replace:

```python
def _run_inference_group(
    net,
    indices: list[tuple],
    obs_flat: np.ndarray,
    all_masks: np.ndarray | None,
    hidden_states: dict,
    out_actions: np.ndarray,
    out_log_probs: np.ndarray,
    out_values: np.ndarray,
) -> None:
    """Batched inference for one (agent, network) group; writes into output arrays."""
    idx_list = [i[0] for i in indices]
    obs_batch = torch.from_numpy(np.ascontiguousarray(obs_flat[idx_list])).float()
    mask_batch = None
    if all_masks is not None:
        mask_batch = torch.from_numpy(np.ascontiguousarray(all_masks[idx_list])).bool()

    hidden_batch = None
    if net.is_recurrent:
        h_list, c_list = [], []
        for _, env_idx, p in indices:
            h, c = hidden_states.get((env_idx, p), net.initial_hidden(1))
            h_list.append(h)
            c_list.append(c)
        hidden_batch = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1))

    with torch.no_grad():
        actions, log_probs, values, new_hidden = net.act(
            obs_batch, action_mask=mask_batch, hidden=hidden_batch,
        )

    if net.is_recurrent and new_hidden is not None:
        for j, (_, env_idx, p) in enumerate(indices):
            hidden_states[(env_idx, p)] = (
                new_hidden[0][:, j:j + 1, :].clone(),
                new_hidden[1][:, j:j + 1, :].clone(),
            )

    idx_arr = np.asarray(idx_list, dtype=np.intp)
    out_actions[idx_arr] = actions.numpy()
    out_log_probs[idx_arr] = log_probs.numpy().astype(np.float32, copy=False)
    out_values[idx_arr] = values.numpy().astype(np.float32, copy=False)
```

   with:

```python
def _run_inference_group(
    model: PolicyModel,
    indices: list[tuple],
    obs_flat: np.ndarray,
    all_masks: np.ndarray | None,
    slot_states: dict[tuple[int, int], State],
    out_actions: np.ndarray,
    out_log_probs: np.ndarray,
    out_values: np.ndarray,
) -> None:
    """Batched inference for one (agent, network) group; writes into output arrays.

    The per-slot states of the group are concatenated into one batch state and
    the new state is sliced back per slot (each slot keeps leaves ``[1, ...]``).
    """
    idx_list = [i[0] for i in indices]
    obs_batch = torch.from_numpy(np.ascontiguousarray(obs_flat[idx_list])).float()
    mask_batch = None
    if all_masks is not None:
        mask_batch = torch.from_numpy(np.ascontiguousarray(all_masks[idx_list])).bool()

    state = cat_batch([slot_states[(env_idx, p)] for _, env_idx, p in indices])
    out = act(model, obs_batch, state, mask_batch)
    if out.state is not None:
        for j, (_, env_idx, p) in enumerate(indices):
            slot_states[(env_idx, p)] = slice_batch(out.state, j)

    idx_arr = np.asarray(idx_list, dtype=np.intp)
    out_actions[idx_arr] = out.actions.numpy()
    out_log_probs[idx_arr] = out.log_probs.numpy().astype(np.float32, copy=False)
    out_values[idx_arr] = out.values.numpy().astype(np.float32, copy=False)
```

5. Replace:

```python
def _apply_command(cmd, networks_by_agent, network_factories, pending) -> None:
    """Load any new checkpoints into the pool and stash the new slot maps.

    The slot maps are applied per-env at the next episode boundary (so a match
    keeps a consistent assignment for its whole episode).
    """
    for aid, ckpts in cmd.new_checkpoints.items():
        if aid not in networks_by_agent:
            continue
        for ckpt_id, sd in ckpts.items():
            if ckpt_id not in networks_by_agent[aid]:
                net = network_factories[aid]()
                net.load_state_dict(sd)
                net.eval()
                networks_by_agent[aid][ckpt_id] = net
```

   with:

```python
def _apply_command(cmd, models_by_agent, model_factories, pending) -> None:
    """Load any new checkpoints into the pool and stash the new slot maps.

    The slot maps are applied per-env at the next episode boundary (so a match
    keeps a consistent assignment for its whole episode).
    """
    for aid, ckpts in cmd.new_checkpoints.items():
        if aid not in models_by_agent:
            continue
        for ckpt_id, sd in ckpts.items():
            if ckpt_id not in models_by_agent[aid]:
                model = model_factories[aid]()
                model.load_state_dict(sd)
                model.eval()
                models_by_agent[aid][ckpt_id] = model
```

6. Replace:

```python
        behavior_policy_version=policy_version,
    )
    if buffer._has_masks and buffer._action_masks is not None:
        chunk.action_masks = torch.from_numpy(buffer._action_masks.copy())
    if buffer._lstm_h_init is not None:
        chunk.lstm_hidden = (buffer._lstm_h_init, buffer._lstm_c_init)
    return chunk
```

   with:

```python
        behavior_policy_version=policy_version,
        initial_state=buffer.initial_state,
    )
    if buffer._has_masks and buffer._action_masks is not None:
        chunk.action_masks = torch.from_numpy(buffer._action_masks.copy())
    return chunk
```

7. Replace:

```python
        model_factories: dict[str, Callable[[], nn.Module]],
```

   with:

```python
        model_factories: dict[str, Callable[[], PolicyModel]],
```

8. Replace:

```python
        # Per-agent network pools: "latest" receives weight updates; checkpoint
        # networks are frozen opponents.
        self._networks: dict[str, dict[str, nn.Module]] = {}
        self._policy_versions: dict[str, int] = {}
        for aid in self.agent_ids:
            nets: dict[str, nn.Module] = {}
```

   with:

```python
        # Per-agent model pools: "latest" receives weight updates; checkpoint
        # models are frozen opponents.
        self._networks: dict[str, dict[str, PolicyModel]] = {}
        self._policy_versions: dict[str, int] = {}
        for aid in self.agent_ids:
            nets: dict[str, PolicyModel] = {}
```

9. Replace:

```python
        # Initial weights for every agent's latest network.
        self.sync_weights()

        self._any_recurrent = self._compute_any_recurrent()
```

   with:

```python
        # Initial weights for every agent's latest network.
        self.sync_weights()
```

10. Replace:

```python
        self._hidden_states: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        if self._any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    aid = self._slot_agent_map[env_idx][p]
                    net = self._networks[aid][LATEST_NETWORK_ID]
                    if net.is_recurrent:
                        self._hidden_states[(env_idx, p)] = net.initial_hidden(1)
```

   with:

```python
        # Model state of every (env, slot); None for stateless models.
        self._slot_states: dict[tuple[int, int], State] = {
            (env_idx, p): self._initial_slot_state(env_idx, p)
            for env_idx in range(num_envs)
            for p in range(num_players)
        }
```

11. Replace:

```python
        if cmd is not None:
            _apply_command(cmd, self._networks, self._model_factories, self._pending)
            self._any_recurrent = self._compute_any_recurrent()

        # Save the recurrent state at chunk start (before inference).
        if self._any_recurrent:
            for env_idx in range(num_envs):
                for p in range(num_players):
                    buf = self._buffers[env_idx][p]
                    if (self._collect_mask[env_idx][p] and buf.steps == 0
                            and (env_idx, p) in self._hidden_states):
                        h, c = self._hidden_states[(env_idx, p)]
                        buf.set_lstm_init(h, c)
```

   with:

```python
        if cmd is not None:
            _apply_command(cmd, self._networks, self._model_factories, self._pending)

        # Record the model state at chunk start (before inference).
        for env_idx in range(num_envs):
            for p in range(num_players):
                buf = self._buffers[env_idx][p]
                if self._collect_mask[env_idx][p] and buf.steps == 0:
                    buf.set_initial_state(self._slot_states[(env_idx, p)])
```

12. Replace:

```python
                net, indices, obs_flat, all_masks, self._hidden_states,
```

   with:

```python
                net, indices, obs_flat, all_masks, self._slot_states,
```

13. Replace:

```python
                        bootstrap_hidden = self._hidden_states.get((env_idx, player_idx))
                        with torch.no_grad():
                            _, _, bootstrap_val, _ = bootstrap_net.act(
                                next_obs_t, hidden=bootstrap_hidden,
                            )
                        bootstrap_val = 0.0 if done else bootstrap_val.item()
```

   with:

```python
                        bootstrap_out = act(
                            bootstrap_net, next_obs_t, self._slot_states[(env_idx, player_idx)],
                        )
                        bootstrap_val = 0.0 if done else bootstrap_out.values.item()
```

14. Replace:

```python
                self._apply_pending_assignment(env_idx)

                if self._any_recurrent:
                    for p in range(num_players):
                        if (env_idx, p) in self._hidden_states:
                            aid = self._slot_agent_map[env_idx][p]
                            net = self._networks[aid][LATEST_NETWORK_ID]
                            self._hidden_states[(env_idx, p)] = net.initial_hidden(1)
```

   with:

```python
                self._apply_pending_assignment(env_idx)
                for p in range(num_players):
                    self._slot_states[(env_idx, p)] = self._initial_slot_state(env_idx, p)
```

15. Replace:

```python
    def _compute_any_recurrent(self) -> bool:
        return any(
            nets[LATEST_NETWORK_ID].is_recurrent for nets in self._networks.values()
        )
```

   with:

```python
    def _initial_slot_state(self, env_idx: int, p: int) -> State:
        """Episode-start state for the model currently assigned to slot (env_idx, p)."""
        aid = self._slot_agent_map[env_idx][p]
        return self._networks[aid][LATEST_NETWORK_ID].initial_state(1)
```

- [ ] **Step 9: Rename the worker wrapper's factory argument**

In `src/colosseum/worker/rollout_worker.py`:

1. Replace:

```python
from colosseum.envs.base_env import BaseEnv
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
```

   with:

```python
from colosseum.envs.base_env import BaseEnv
from colosseum.networks.model import PolicyModel
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
```

2. Replace:

```python
    network_factories: dict[str, Callable[[], Any]],
```

   with:

```python
    model_factories: dict[str, Callable[[], PolicyModel]],
```

3. Replace:

```python
        model_factories=network_factories,
```

   with:

```python
        model_factories=model_factories,
```

- [ ] **Step 10: Build models with `build_model` in the launcher and in distributed mode**

In `src/colosseum/launcher.py`:

1. Replace:

```python
def _create_network(config: ColosseumConfig):
    """Create ActorCriticNetwork inside a worker/learner process."""
    from colosseum.core.registry import build_network
    return build_network(config)
```

   with:

```python
def _create_model(config: ColosseumConfig):
    """Create the agent's PolicyModel inside a worker/learner process."""
    from colosseum.core.registry import build_model
    return build_model(config)
```

2. Replace:

```python
    Builds per-agent network factories INSIDE the process to avoid
```

   with:

```python
    Builds per-agent model factories INSIDE the process to avoid
```

3. Replace:

```python
    # Build per-agent network factories inside this process (pickling safe)
    network_factories = {}
    for aid in agent_ids:
        acfg = agent_configs[aid]
        # Default-arg capture ensures each lambda gets its own config
        network_factories[aid] = lambda _cfg=acfg: _create_network(_cfg)
```

   with:

```python
    # Build per-agent model factories inside this process (pickling safe)
    model_factories = {}
    for aid in agent_ids:
        acfg = agent_configs[aid]
        # Default-arg capture ensures each lambda gets its own config
        model_factories[aid] = lambda _cfg=acfg: _create_model(_cfg)
```

4. Replace:

```python
        network_factories=network_factories,
```

   with:

```python
        model_factories=model_factories,
```

5. Replace:

```python
        net = _create_network(config)
        kickstart = None
```

   with:

```python
        model = _create_model(config)
        kickstart = None
```

6. Replace:

```python
            teacher = _create_network(config)
```

   with:

```python
            teacher = _create_model(config)
```

7. Replace:

```python
        return algo_cls(net, config.algorithm, **kwargs)
```

   with:

```python
        return algo_cls(model, config.algorithm, **kwargs)
```

In `src/colosseum/distributed.py`:

1. Replace:

```python
    from colosseum.core.registry import build_network, import_class
```

   with:

```python
    from colosseum.core.registry import build_model, import_class
```

2. Replace:

```python
        net = build_network(acfg)
        kickstart = None
```

   with:

```python
        model = build_model(acfg)
        kickstart = None
```

3. Replace:

```python
            teacher = build_network(acfg)
```

   with:

```python
            teacher = build_model(acfg)
```

4. Replace:

```python
        return algo_cls(net, acfg.algorithm, **kwargs)
```

   with:

```python
        return algo_cls(model, acfg.algorithm, **kwargs)
```

5. Replace:

```python
    from colosseum.core.registry import build_network
```

   with:

```python
    from colosseum.core.registry import build_model
```

6. Replace:

```python
    network_factories = {
        aid: partial(build_network, agent_configs[aid]) for aid in agent_ids
    }
```

   with:

```python
    model_factories = {
        aid: partial(build_model, agent_configs[aid]) for aid in agent_ids
    }
```

7. Replace:

```python
        network_factories=network_factories,
```

   with:

```python
        model_factories=model_factories,
```

Check: `grep -rn "build_network\|_create_network\|network_factories" src` prints only `src/colosseum/cli.py` (two BC/eval uses, migrated in T1.7), `src/colosseum/eval.py` (migrated in T1.7) and the transitional `build_network` definition in `src/colosseum/core/registry.py`.

- [ ] **Step 11: Migrate the existing tests to `PolicyModel`**

In `tests/unit/test_appo.py`:

1. Replace:

```python
from colosseum.networks.actor_critic import ActorCriticNetwork
from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue


def _make_network():
    return ActorCriticNetwork(TicTacToeEncoder(), TicTacToePolicy(), TicTacToeValue())
```

   with:

```python
from helpers import make_ttt_model


def _make_network():
    return make_ttt_model()
```

In `tests/unit/test_bc.py`:

1. Replace:

```python
from colosseum.networks.actor_critic import ActorCriticNetwork


def _make_network():
    """Create a small TicTacToe-like network for testing."""
    from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue
    return ActorCriticNetwork(TicTacToeEncoder(), TicTacToePolicy(), TicTacToeValue())
```

   with:

```python
from helpers import make_ttt_model


def _make_network():
    """Create a small TicTacToe model for testing."""
    return make_ttt_model()
```

In `tests/unit/test_composite_actions.py`:

1. Replace:

```python
    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
```

   with:

```python
    from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.cores import NoCore
```

2. Replace:

```python
    net = ActorCriticNetwork(Enc(), Pol(), Val())
```

   with:

```python
    net = ComposedModel(Enc(), NoCore(LATENT), Pol(), Val())
```

In `tests/unit/test_action_masking.py`:

1. Replace:

```python
    # to_device should preserve masks
    chunk2 = chunk.to_device("cpu")
```

   with:

```python
    # to() should preserve masks
    chunk2 = chunk.to("cpu")
```

2. Replace:

```python
    assert chunk.action_masks is None
    chunk2 = chunk.to_device("cpu")
```

   with:

```python
    assert chunk.action_masks is None
    chunk2 = chunk.to("cpu")
```

3. Replace:

```python
    net = _make_network(obs_dim=4, num_actions=3)
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
```

   with:

```python
    from helpers import make_simple_model

    net = make_simple_model(obs_dim=4, hidden_dim=32, num_actions=3)
    config = AlgorithmConfig(name="appo", num_epochs=1, minibatch_chunks=0)
```

In `tests/unit/test_review_fixes.py`:

1. Replace:

```python
    from helpers import make_simple_network

    def factory():
        return make_simple_network(obs_dim=4, num_actions=3)
```

   with:

```python
    from helpers import make_simple_model

    def factory():
        return make_simple_model(obs_dim=4, num_actions=3)
```

In `tests/unit/test_performance.py` (two APPO tests):

```bash
sed -i 's/    from helpers import make_simple_network/    from helpers import make_simple_model/; s/    net = make_simple_network(obs_dim=4, hidden_dim=32, num_actions=3)/    net = make_simple_model(obs_dim=4, hidden_dim=32, num_actions=3)/' tests/unit/test_performance.py
```

In `tests/unit/test_recurrent.py` delete the sections that tested the removed APPO/chunk/buffer recurrent paths (their replacements are `tests/unit/test_appo_unroll.py` and `tests/contract/test_rollout_state.py`):

```bash
.venv/bin/python /tmp/drop_sections.py tests/unit/test_recurrent.py \
  "APPO recurrent training" "TrajectoryChunk lstm_hidden support" "RolloutBuffer LSTM init storage"
```

(`/tmp/drop_sections.py` is the helper from T0.2 Step 10; recreate it from there if `/tmp` was cleaned.) The remaining `test_recurrent.py` tests exercise `ActorCriticNetwork` and the transitional `build_network`; they are deleted with them in T1.7.

Check: `grep -rn "lstm_hidden\|set_lstm_init\|\.to_device(" src tests` prints nothing.

- [ ] **Step 12: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/test_appo_unroll.py tests/contract -v`

Expected: all pass (`test_appo_unroll.py`: 10, `test_rollout_state.py`: 5, characterization: 9).

- [ ] **Step 13: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!` (the autofix removes imports that became unused in `test_recurrent.py`, `test_appo.py`, `test_bc.py`); all tests pass.

- [ ] **Step 14: Add the recurrent end-to-end pipeline test and run all process-level tests**

Append to `tests/integration/test_pipelines.py`:

```python


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_full_pipeline_lstm_core(tmp_path):
    """Recurrent end-to-end run (R1-01): worker chunks with LSTM state train in the learner."""
    from colosseum.launcher import Launcher

    config = _config(
        "tic_tac_toe.yaml", tmp_path,
        training={"total_timesteps": 3000},
        rollout={"num_workers": 1, "envs_per_worker": 2, "chunk_length": 8},
        learner={"batch_chunks": 2, "queue_size": 16},
    )
    data = config.model_dump()
    data["networks"]["core"] = {"class": "colosseum.networks.cores.LSTMCore", "kwargs": {"hidden_size": 32}}
    Launcher(ColosseumConfig(**data)).launch()
    assert CheckpointManager(str(tmp_path / "checkpoints")).list_checkpoints("agent_0")
```

Run: `.venv/bin/python -m pytest tests/integration -m slow -v`

Expected: all pass, including `test_full_pipeline_lstm_core` (before this task an LSTM run crashed the learner on its first batch, R1-01).

- [ ] **Step 15: Commit**

```bash
git add src tests
git commit -m "feat: train through PolicyModel.unroll; chunks carry initial_state; worker keeps per-slot State"
```

---

### Task T1.6: Contract test: the learner reproduces the worker's log-probs and values for four cores

Spec §3, criterion 2, first bullet. The `RolloutLoop` state handling itself landed in T1.5 (it had to switch together with the chunk format; see "Contract notes"); this task locks it in with the contract test through the real producers: chunks come from the real `RolloutLoop` driven in-process and are re-evaluated by the real `APPO` at identical weights.

**Files:**
- Modify: `tests/contract/harness.py` (rewrite: adds `learner_eval`)
- Test: `tests/contract/test_worker_learner_consistency.py`

**Interfaces:**
- Consumes: T1.5 (`APPO.evaluate_chunks`, `RolloutLoop` with `PolicyModel` factories, `TrajectoryChunk.initial_state`), T0.5 harness (`make_loop`, `run_until_chunks`, `weights_payload`), T1.4 helpers (`CORE_KINDS`).
- Produces:
  - `tests/contract/harness.py::learner_eval(model, chunks) -> (learner_log_probs, learner_values, worker_log_probs, worker_values)` — all `[T*B]` time-major; wraps `APPO(model, AlgorithmConfig(), device="cpu").evaluate_chunks(chunks)`. Later contract tests (T3.x, T4.x) reuse it.
  - Test `test_learner_reproduces_worker_logprobs_and_values[none|lstm|gru|window]`: 2 envs × 2 seats, `chunk_length=8`, episodes of 5 steps (so chunks span episode boundaries and start mid-episode with a non-zero state), max abs diff < 1e-5.
  - Test `test_heterogeneous_agents_and_checkpoint_opponents`: agents with different cores (LSTM, window attention) share envs, one seat is a frozen stateful checkpoint; per-agent reproduction < 1e-5.

- [ ] **Step 1: Add `learner_eval` to the harness**

Replace the whole of `tests/contract/harness.py` with:

```python
"""Helpers to drive the real ``RolloutLoop`` in-process (no worker processes)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
from torch import Tensor

from colosseum.algorithms.appo import APPO
from colosseum.core.config import AlgorithmConfig
from colosseum.core.types import MatchResult, TrajectoryChunk, WeightPayload, WorkerCommand
from colosseum.worker.rollout_loop import LoopIO, RolloutLoop
from helpers import CountingEnv, make_simple_model

OBS_DIM = 4
NUM_ACTIONS = 3


def simple_factory(core: str = "none") -> Any:
    """Model factory used by the contract tests: SimpleEncoder -> core -> heads."""
    return make_simple_model(obs_dim=OBS_DIM, hidden_dim=16, num_actions=NUM_ACTIONS, core=core)


def counting_env_fn(num_players: int = 2, episode_length: int = 5) -> Callable[[], CountingEnv]:
    return partial(CountingEnv, num_players=num_players, episode_length=episode_length,
                   num_actions=NUM_ACTIONS, obs_dim=OBS_DIM)


def weights_payload(agent_id: str, model: Any, version: int) -> WeightPayload:
    """Weight payload carrying ``model``'s current parameters."""
    return WeightPayload(
        agent_id=agent_id,
        policy_version=version,
        state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
    )


@dataclass
class LoopRecorder:
    """In-memory endpoints for ``LoopIO``: collects outputs, serves queued inputs."""

    chunks: list[TrajectoryChunk] = field(default_factory=list)
    results: list[MatchResult] = field(default_factory=list)
    pending_weights: dict[str, WeightPayload] = field(default_factory=dict)
    pending_commands: list[WorkerCommand] = field(default_factory=list)

    def poll_weights(self, agent_id: str) -> WeightPayload | None:
        return self.pending_weights.pop(agent_id, None)

    def poll_command(self) -> WorkerCommand | None:
        return self.pending_commands.pop(0) if self.pending_commands else None

    def io(self) -> LoopIO:
        return LoopIO(
            send_chunk=self.chunks.append,
            poll_weights=self.poll_weights,
            report_result=self.results.append,
            poll_command=self.poll_command,
        )


def make_loop(
    *,
    agent_ids: list[str] | None = None,
    model_factories: dict[str, Callable[[], Any]] | None = None,
    env_fn: Callable[[], Any] | None = None,
    num_envs: int = 2,
    chunk_length: int = 4,
    seed: int = 123,
    initial_weights: dict[str, WeightPayload] | None = None,
    **loop_kwargs: Any,
) -> tuple[RolloutLoop, LoopRecorder]:
    """Build a ``RolloutLoop`` wired to a fresh ``LoopRecorder``."""
    agent_ids = agent_ids or ["agent_0"]
    if model_factories is None:
        model_factories = {aid: simple_factory for aid in agent_ids}
    rec = LoopRecorder()
    if initial_weights:
        rec.pending_weights.update(initial_weights)
    loop = RolloutLoop(
        worker_id=0,
        env_fn=env_fn or counting_env_fn(),
        num_envs=num_envs,
        chunk_length=chunk_length,
        agent_ids=agent_ids,
        model_factories=model_factories,
        io=rec.io(),
        seed=seed,
        **loop_kwargs,
    )
    return loop, rec


def run_steps(loop: RolloutLoop, n: int) -> None:
    for _ in range(n):
        loop.step()


def run_until_chunks(loop: RolloutLoop, rec: LoopRecorder, n_chunks: int,
                     max_steps: int = 10_000) -> list[TrajectoryChunk]:
    """Step until at least ``n_chunks`` chunks were sent; return the first ``n_chunks``."""
    for _ in range(max_steps):
        if len(rec.chunks) >= n_chunks:
            return rec.chunks[:n_chunks]
        loop.step()
    raise AssertionError(f"only {len(rec.chunks)} chunks after {max_steps} steps")


def step_index(chunk: TrajectoryChunk, episode_length: int = 5) -> list[int]:
    """In-episode step index of every transition (decoded from CountingEnv obs)."""
    return [round(float(x) * episode_length) for x in chunk.observations[:, 0]]


def player_index(chunk: TrajectoryChunk) -> set[int]:
    """Set of player indices whose observations appear in ``chunk``."""
    return {round(float(x)) for x in chunk.observations[:, 1]}


def learner_eval(model: Any, chunks: list[TrajectoryChunk]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Re-evaluate ``chunks`` with a real APPO around ``model``.

    Returns ``(learner_log_probs, learner_values, worker_log_probs, worker_values)``,
    all ``[T*B]`` in time-major order (index ``t*B + b`` for chunk ``b``).
    """
    algo = APPO(model, AlgorithmConfig(), device="cpu")
    lp, v = algo.evaluate_chunks(chunks)
    worker_lp = torch.stack([c.action_log_probs for c in chunks], dim=1).reshape(-1)
    worker_v = torch.stack([c.values for c in chunks], dim=1).reshape(-1)
    return lp, v, worker_lp, worker_v
```

- [ ] **Step 2: Write the contract test**

Create `tests/contract/test_worker_learner_consistency.py`:

```python
"""Contract: the learner reproduces the worker's log-probs and values at equal weights.

Chunks come from the real RolloutLoop (in-process) and are re-evaluated by the
real APPO (``model.unroll`` from ``cat_batch(chunk.initial_state)``).
"""

from __future__ import annotations

import pytest
import torch

from harness import learner_eval, make_loop, run_until_chunks, simple_factory, weights_payload
from helpers import CORE_KINDS

TOL = 1e-5


def _max_diff(a, b) -> float:
    return float((a - b).abs().max())


@pytest.mark.parametrize("core", CORE_KINDS)
def test_learner_reproduces_worker_logprobs_and_values(core):
    torch.manual_seed(0)
    learner_model = simple_factory(core)
    loop, rec = make_loop(
        model_factories={"agent_0": lambda: simple_factory(core)},
        num_envs=2, chunk_length=8,  # episodes last 5 steps -> chunks span boundaries
        initial_weights={"agent_0": weights_payload("agent_0", learner_model, 0)},
    )
    chunks = run_until_chunks(loop, rec, 8)  # two chunks per slot
    loop.close()

    assert any(bool(c.dones[:-1].any()) for c in chunks), "no chunk spans an episode boundary"
    if core != "none":
        assert any(
            float(sum(leaf.float().abs().sum() for leaf in c.initial_state.values())) > 0
            for c in chunks
        ), "no chunk starts mid-episode with a non-zero state"

    lp, v, worker_lp, worker_v = learner_eval(learner_model, chunks)
    assert _max_diff(lp, worker_lp) < TOL
    assert _max_diff(v, worker_v) < TOL


def test_heterogeneous_agents_and_checkpoint_opponents():
    """Two agents with different cores share envs; a frozen stateful checkpoint plays too."""
    torch.manual_seed(0)
    model_a = simple_factory("lstm")
    model_b = simple_factory("window")
    ckpt_b = {k: v.clone() for k, v in simple_factory("window").state_dict().items()}
    loop, rec = make_loop(
        agent_ids=["a", "b"],
        model_factories={"a": lambda: simple_factory("lstm"), "b": lambda: simple_factory("window")},
        num_envs=3, chunk_length=8,
        slot_agent_map=[["a", "b"], ["b", "a"], ["a", "b"]],
        slot_network_map=[["latest", "latest"], ["latest", "latest"], ["latest", "ckpt_v1"]],
        collect_mask=[[True, True], [True, True], [True, False]],
        checkpoint_state_dicts_by_agent={"a": {}, "b": {"ckpt_v1": ckpt_b}},
        initial_weights={
            "a": weights_payload("a", model_a, 0),
            "b": weights_payload("b", model_b, 0),
        },
    )
    run_until_chunks(loop, rec, 10)
    loop.close()

    for agent_id, model in (("a", model_a), ("b", model_b)):
        chunks = [c for c in rec.chunks if c.agent_id == agent_id]
        assert len(chunks) >= 2
        lp, v, worker_lp, worker_v = learner_eval(model, chunks)
        assert _max_diff(lp, worker_lp) < TOL, agent_id
        assert _max_diff(v, worker_v) < TOL, agent_id
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python -m pytest tests/contract/test_worker_learner_consistency.py -v`

Expected: 5 passed (the implementation is from T1.5). Measured differences are around 1e-7 (window attention in the worker runs the fused eval-mode attention kernel, the learner the training-mode path).

- [ ] **Step 4: Prove the test catches a wrong `initial_state`**

Temporarily break the chunk-start state in `src/colosseum/worker/rollout_loop.py`: replace

```python
                    buf.set_initial_state(self._slot_states[(env_idx, p)])
```

with

```python
                    buf.set_initial_state(self._initial_slot_state(env_idx, p))
```

Run: `.venv/bin/python -m pytest tests/contract/test_worker_learner_consistency.py -q`

Expected: the `lstm`, `gru`, `window` cases and `test_heterogeneous_agents_and_checkpoint_opponents` FAIL (`no chunk starts mid-episode with a non-zero state` / reproduction assertion). Then restore the file: `git checkout src/colosseum/worker/rollout_loop.py`, and re-run Step 3 (5 passed).

- [ ] **Step 5: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!`; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/contract
git commit -m "test: contract - learner reproduces worker log-probs and values for none/LSTM/GRU/attention cores"
```

---

### Task T1.7: Eval, BC and kickstart on `PolicyModel`; delete `actor_critic.py`

Findings: R4-11 / ET-10 (eval crashed on recurrent agents: no per-slot state), R1-02 (kickstart and BC bypassed the core). This is the minimal migration; the full rewrites are T4.3 (kickstart: forward KL, masks, unroll), T4.4 (BC: sequence training, distribution-aware loss) and T7.x (eval engine and statistics).

**Files:**
- Modify: `src/colosseum/eval.py` (`evaluate_agents`, `_run_matches`), `src/colosseum/bc/kickstart.py`, `src/colosseum/bc/offline_bc.py`, `src/colosseum/cli.py` (`bc`, `eval` commands), `src/colosseum/core/registry.py` (delete `build_network`)
- Delete: `src/colosseum/networks/actor_critic.py`, `tests/unit/test_recurrent.py`
- Modify (tests): `tests/helpers.py`, `tests/unit/test_eval.py`, `tests/unit/test_bc.py`, `tests/unit/test_review_fixes.py`, `tests/unit/test_action_masking.py`

**Interfaces:**
- Consumes: T1.2 (`PolicyModel`, `act`), T1.1 (`cat_batch`, `slice_batch`, `State`), T1.4 (`build_model`, helpers), T1.5 (training path already on `PolicyModel`).
- Produces:
  - `evaluate_agents(agent_configs, env_fn, model_factory: Callable[[], PolicyModel], num_matches=100, num_envs=8, model_factories: dict[str, Callable[[], PolicyModel]] | None = None, deterministic=False) -> EvalMatrix` (parameters renamed from `network_factory`/`network_factories`). `_run_matches` keeps one `State` per (env, slot) for the agent playing it, batches it per agent with `cat_batch`/`slice_batch` around `act(...)`, and resets every slot of an env to `initial_state(1)` of its newly rolled agent when the match ends. Result semantics are unchanged (T7.x changes them).
  - `OfflineBCTrainer(model: PolicyModel, lr=1e-3, device="cpu", action_type="discrete")`; property `model` (was `network`); the loss uses `model.step(obs, model.initial_state(B, obs.device)).dist`.
  - `KickstartLoss(teacher_model: PolicyModel, initial_lambda=1.0, decay_steps=50000)`; `compute(student_model: PolicyModel, observations) -> Tensor` (scaled `KL(student || teacher)` from both models' `step` at their initial state; still reverse KL and mask-free until T4.3).
  - CLI `colosseum bc` and `colosseum eval` build models with `build_model`.
  - Removed for good: `colosseum.networks.actor_critic`, `ActorCriticNetwork`, `build_network`, `is_recurrent`, `initial_hidden`, `evaluate_actions`, `evaluate_actions_recurrent`, `tests/helpers.make_simple_network`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_eval.py` replace:

```python
from colosseum.networks.actor_critic import ActorCriticNetwork
from examples.tic_tac_toe.env import TicTacToeEnv
from examples.tic_tac_toe.networks import TicTacToeEncoder, TicTacToePolicy, TicTacToeValue


def _make_net():
    return ActorCriticNetwork(TicTacToeEncoder(), TicTacToePolicy(), TicTacToeValue())
```

with:

```python
from examples.tic_tac_toe.env import TicTacToeEnv
from helpers import CountingEnv, make_simple_model, make_ttt_model


def _make_net():
    return make_ttt_model()
```

rename the keyword in the existing calls (`sed -i 's/        network_factory=_make_net,/        model_factory=_make_net,/' tests/unit/test_eval.py`) and append:

```python
def test_evaluate_stateful_models():
    """Eval tracks per-slot model state (LSTM vs window attention) and resets it per match."""
    from functools import partial

    def lstm():
        return make_simple_model(obs_dim=4, num_actions=3, core="lstm")

    def window():
        return make_simple_model(obs_dim=4, num_actions=3, core="window")

    agent_configs = {
        "lstm": {"state_dict": lstm().state_dict()},
        "window": {"state_dict": window().state_dict()},
    }
    matrix = evaluate_agents(
        agent_configs,
        env_fn=partial(CountingEnv, num_players=2, episode_length=5, num_actions=3, obs_dim=4),
        model_factory=lstm,
        model_factories={"lstm": lstm, "window": window},
        num_matches=12,
        num_envs=3,
    )
    r = matrix.get("lstm", "window")
    assert r is not None and r.num_matches == 12
    assert r.wins_a + r.wins_b + r.draws == 12
```

Append to `tests/unit/test_bc.py`:

```python
def test_offline_bc_and_kickstart_accept_stateful_models():
    """BC and kickstart go through PolicyModel.step, so a core with output_dim != latent_dim works."""
    from helpers import make_simple_model

    def lstm_model():
        return make_simple_model(obs_dim=8, num_actions=4, core="lstm")

    trainer = OfflineBCTrainer(lstm_model(), lr=1e-3, action_type="discrete")
    trainer.add_data(torch.randn(32, 8), torch.randint(0, 4, (32,)))
    assert trainer.train(num_epochs=1, batch_size=16)["bc_loss"] > 0
    assert trainer.model is not None

    ks = KickstartLoss(lstm_model(), initial_lambda=1.0, decay_steps=10)
    loss = ks.compute(lstm_model(), torch.randn(5, 8))
    assert loss.shape == () and torch.isfinite(loss) and loss.item() >= 0
```

- [ ] **Step 2: Run them and see them fail**

Run: `.venv/bin/python -m pytest tests/unit/test_eval.py tests/unit/test_bc.py -v`

Expected: `test_eval.py` tests calling `evaluate_agents` fail with `TypeError: evaluate_agents() got an unexpected keyword argument 'model_factory'`; `test_offline_bc_and_kickstart_accept_stateful_models` fails with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x16 and 24x4)` (BC skipped the core).

- [ ] **Step 3: Eval on `PolicyModel` with per-slot state**

In `src/colosseum/eval.py`:

1. Replace:

```python
from colosseum.networks.actor_critic import ActorCriticNetwork
```

   with:

```python
from colosseum.networks.model import PolicyModel, act
from colosseum.networks.state import State, cat_batch, slice_batch
```

2. Replace:

```python
    network_factory: Callable[[], ActorCriticNetwork],
    num_matches: int = 100,
    num_envs: int = 8,
    network_factories: dict[str, Callable[[], ActorCriticNetwork]] | None = None,
```

   with:

```python
    model_factory: Callable[[], PolicyModel],
    num_matches: int = 100,
    num_envs: int = 8,
    model_factories: dict[str, Callable[[], PolicyModel]] | None = None,
```

3. Replace:

```python
        network_factory: Default factory to create an ActorCriticNetwork.
        num_matches: Number of matches to play.
        num_envs: Number of parallel environments for evaluation.
        network_factories: Optional per-agent network factories. If provided,
            overrides ``network_factory`` for the corresponding agent.
```

   with:

```python
        model_factory: Default factory to create a PolicyModel.
        num_matches: Number of matches to play.
        num_envs: Number of parallel environments for evaluation.
        model_factories: Optional per-agent model factories. If provided,
            overrides ``model_factory`` for the corresponding agent.
```

4. Replace:

```python
    # Load all networks
    agents: list[tuple[str, ActorCriticNetwork]] = []
    for aid in agent_ids:
        cfg = agent_configs[aid]
        factory = (
            network_factories[aid]
            if network_factories and aid in network_factories
            else network_factory
        )
```

   with:

```python
    # Load all models
    agents: list[tuple[str, PolicyModel]] = []
    for aid in agent_ids:
        cfg = agent_configs[aid]
        factory = (
            model_factories[aid]
            if model_factories and aid in model_factories
            else model_factory
        )
```

5. Replace:

```python
def _run_matches(
    agents: list[tuple[str, ActorCriticNetwork]],
```

   with:

```python
def _run_matches(
    agents: list[tuple[str, PolicyModel]],
```

6. Replace:

```python
    ``info["outcome"]``) over cumulative reward.
```

   with:

```python
    ``info["outcome"]``) over cumulative reward. Every (env, slot) keeps the
    model state of the agent playing it; the state is reset when the match ends.
```

7. Replace:

```python
        agents: List of (agent_id, network) tuples.
```

   with:

```python
        agents: List of (agent_id, model) tuples.
```

8. Replace:

```python
    # Per-env slot→agent assignment, re-rolled on each match boundary.
    slot_assign: list[list[int]] = [_roll_assignment() for _ in range(actual_envs)]
```

   with:

```python
    # Per-env slot→agent assignment, re-rolled on each match boundary.
    slot_assign: list[list[int]] = [_roll_assignment() for _ in range(actual_envs)]

    def _fresh_state(e: int, p: int) -> State:
        return agents[slot_assign[e][p]][1].initial_state(1)

    slot_states: dict[tuple[int, int], State] = {
        (e, p): _fresh_state(e, p) for e in range(actual_envs) for p in range(num_players)
    }
```

9. Replace:

```python
        for agent_idx, slot_pairs in groups.items():
            net = agents[agent_idx][1]
            flat_idxs = [e * num_players + p for (e, p) in slot_pairs]
            obs_batch = torch.tensor(obs_flat[flat_idxs], dtype=torch.float32)
            mask_batch = None
            if masks is not None:
                mask_batch = torch.tensor(masks[flat_idxs], dtype=torch.bool)
            with torch.no_grad():
                actions, _, _, _ = net.act(
                    obs_batch, action_mask=mask_batch, deterministic=deterministic,
                )
            actions_arr = actions.numpy()
            for k, (e, p) in enumerate(slot_pairs):
                actions_np[e, p] = actions_arr[k]
```

   with:

```python
        for agent_idx, slot_pairs in groups.items():
            model = agents[agent_idx][1]
            flat_idxs = [e * num_players + p for (e, p) in slot_pairs]
            obs_batch = torch.tensor(obs_flat[flat_idxs], dtype=torch.float32)
            mask_batch = None
            if masks is not None:
                mask_batch = torch.tensor(masks[flat_idxs], dtype=torch.bool)
            state = cat_batch([slot_states[(e, p)] for (e, p) in slot_pairs])
            out = act(model, obs_batch, state, mask_batch, deterministic=deterministic)
            actions_arr = out.actions.numpy()
            for k, (e, p) in enumerate(slot_pairs):
                actions_np[e, p] = actions_arr[k]
                if out.state is not None:
                    slot_states[(e, p)] = slice_batch(out.state, k)
```

10. Replace:

```python
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0
                slot_assign[env_idx] = _roll_assignment()
```

   with:

```python
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0
                slot_assign[env_idx] = _roll_assignment()
                for p in range(num_players):
                    slot_states[(env_idx, p)] = _fresh_state(env_idx, p)
```

- [ ] **Step 4: Kickstart on `PolicyModel`**

In `src/colosseum/bc/kickstart.py`:

1. Replace:

```python
from colosseum.networks.actor_critic import ActorCriticNetwork
```

   with:

```python
from colosseum.networks.model import PolicyModel
```

2. Replace:

```python
        teacher_network: ActorCriticNetwork,
```

   with:

```python
        teacher_model: PolicyModel,
```

3. Replace:

```python
            teacher_network: Frozen BC model (will be set to eval, no grad).
```

   with:

```python
            teacher_model: Frozen BC model (will be set to eval, no grad).
```

4. Replace:

```python
        self._teacher = teacher_network
```

   with:

```python
        self._teacher = teacher_model
```

5. Replace:

```python
        student_network: ActorCriticNetwork,
        observations: torch.Tensor,
```

   with:

```python
        student_model: PolicyModel,
        observations: torch.Tensor,
```

6. Replace:

```python
            student_network: The student policy being trained.
```

   with:

```python
            student_model: The student policy being trained.
```

7. Replace:

```python
        # Get student distribution
        student_latent = student_network.encoder(observations)
        student_dist = student_network.policy(student_latent)

        # Get teacher distribution (no grad)
        with torch.no_grad():
            teacher_latent = self._teacher.encoder(observations)
            teacher_dist = self._teacher.policy(teacher_latent)
```

   with:

```python
        # Both policies score every observation from their initial state
        # (sequence-aware kickstarting comes with the kickstart rewrite).
        batch, device = observations.shape[0], observations.device
        student_dist = student_model.step(observations, student_model.initial_state(batch, device)).dist
        with torch.no_grad():
            teacher_dist = self._teacher.step(observations, self._teacher.initial_state(batch, device)).dist
```

- [ ] **Step 5: Offline BC on `PolicyModel`**

In `src/colosseum/bc/offline_bc.py`:

1. Replace:

```python
from colosseum.networks.actor_critic import ActorCriticNetwork
```

   with:

```python
from colosseum.networks.model import PolicyModel
```

2. Replace:

```python
        trainer = OfflineBCTrainer(network, lr=1e-3)
```

   with:

```python
        trainer = OfflineBCTrainer(model, lr=1e-3)
```

3. Replace:

```python
    Scope/limitations: BC trains the feedforward policy path (encoder -> policy
    head); it does not unroll a recurrent trunk, and it loads the full dataset
    into memory. For very large replay corpora or recurrent policies, pre-train
    feedforward then fine-tune with RL, or stream data in shards via repeated
    ``add_data`` + ``train`` calls.
```

   with:

```python
    Scope/limitations: every sample is scored with ``model.step`` from the
    model's initial state (no sequence unroll yet), and the full dataset is
    loaded into memory. Stream very large corpora in shards via repeated
    ``add_data`` + ``train`` calls.
```

4. Replace:

```python
        network: ActorCriticNetwork,
        lr: float = 1e-3,
```

   with:

```python
        model: PolicyModel,
        lr: float = 1e-3,
```

5. Replace:

```python
            network: The actor-critic network to train (only encoder+policy used).
```

   with:

```python
            model: The PolicyModel to train (only its policy distribution is used).
```

6. Replace:

```python
        self._network = network.to(device)
        self._device = device
        self._action_type = action_type
        self._optimizer = torch.optim.Adam(network.parameters(), lr=lr)
```

   with:

```python
        self._model = model.to(device)
        self._device = device
        self._action_type = action_type
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
```

7. Replace:

```python
    @property
    def network(self) -> ActorCriticNetwork:
        return self._network
```

   with:

```python
    @property
    def model(self) -> PolicyModel:
        return self._model
```

8. Replace:

```python
        latent = self._network.encoder(obs)
        dist = self._network.policy(latent)
```

   with:

```python
        dist = self._model.step(obs, self._model.initial_state(obs.shape[0], obs.device)).dist
```

- [ ] **Step 6: CLI `bc` and `eval` build models with `build_model`**

In `src/colosseum/cli.py`:

1. Replace:

```python
    from colosseum.core.registry import build_network

    logging.basicConfig(
```

   with:

```python
    from colosseum.core.registry import build_model

    logging.basicConfig(
```

2. Replace:

```python
    # Build network
    network = build_network(cfg)
```

   with:

```python
    model = build_model(cfg)
```

3. Replace:

```python
    trainer = OfflineBCTrainer(
        network=network,
```

   with:

```python
    trainer = OfflineBCTrainer(
        model=model,
```

4. Replace:

```python
    torch.save(network.state_dict(), output)
```

   with:

```python
    torch.save(model.state_dict(), output)
```

5. Replace:

```python
    from colosseum.core.registry import build_network, import_class
```

   with:

```python
    from colosseum.core.registry import build_model, import_class
```

6. Replace:

```python
    def network_factory():
        return build_network(cfg)

    matrix = evaluate_agents(
        agent_configs, env_fn, network_factory,
```

   with:

```python
    def model_factory():
        return build_model(cfg)

    matrix = evaluate_agents(
        agent_configs, env_fn, model_factory,
```

- [ ] **Step 7: Run the failing tests again**

Run: `.venv/bin/python -m pytest tests/unit/test_eval.py tests/unit/test_bc.py -v`

Expected: all pass (`test_eval.py`: 7, `test_bc.py`: 8).

- [ ] **Step 8: Delete `ActorCriticNetwork`, the transitional `build_network` and the tests that only covered them**

In `src/colosseum/core/registry.py` replace:

```python
if TYPE_CHECKING:
    from colosseum.core.config import ColosseumConfig
    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.model import PolicyModel
```

with:

```python
if TYPE_CHECKING:
    from colosseum.core.config import ColosseumConfig
    from colosseum.networks.model import PolicyModel
```

and delete the whole function:

```python
def build_network(config: ColosseumConfig) -> ActorCriticNetwork:
    """TRANSITIONAL: the legacy ``ActorCriticNetwork`` built from the new ``networks`` schema.

    Kept only until every caller uses :func:`build_model`; deleted together with
    ``networks/actor_critic.py``. Supports ``core`` = null, ``NoCore``,
    ``LSTMCore`` and ``GRUCore``.
    """
    import torch.nn as nn

    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.cores import GRUCore, LSTMCore, NoCore

    net = config.networks
    if net.model_class:
        raise ConfigError("networks.model_class is not supported by the legacy ActorCriticNetwork path")
    encoder = import_class(net.encoder_class)(**net.kwargs)
    out_dim = encoder.latent_dim
    recurrent = None
    if net.core is not None:
        core_cls = import_class(net.core.class_path)
        hidden = int(net.core.kwargs.get("hidden_size", 128))
        layers = int(net.core.kwargs.get("num_layers", 1))
        if issubclass(core_cls, LSTMCore):
            recurrent = nn.LSTM(out_dim, hidden, layers)
            out_dim = hidden
        elif issubclass(core_cls, GRUCore):
            recurrent = nn.GRU(out_dim, hidden, layers)
            out_dim = hidden
        elif not issubclass(core_cls, NoCore):
            raise ConfigError(f"core {net.core.class_path!r} is not supported by the legacy ActorCriticNetwork path")
    policy = _build_head(net.policy_class, out_dim, net.kwargs)
    value = _build_head(net.value_class, out_dim, net.kwargs)
    return ActorCriticNetwork(encoder, policy, value, recurrent=recurrent)
```

In `tests/helpers.py` delete:

```python
def make_simple_network(obs_dim=8, hidden_dim=16, num_actions=4):
    """Legacy feedforward ActorCriticNetwork (deleted together with actor_critic.py)."""
    from colosseum.networks.actor_critic import ActorCriticNetwork

    encoder = SimpleEncoder(obs_dim, hidden_dim)
    policy = SimplePolicy(hidden_dim, num_actions)
    value = SimpleValue(hidden_dim)
    return ActorCriticNetwork(encoder, policy, value)
```

Then:

```bash
git rm -q src/colosseum/networks/actor_critic.py tests/unit/test_recurrent.py
sed -i '/^from colosseum.networks.actor_critic import ActorCriticNetwork$/d' tests/unit/test_review_fixes.py
.venv/bin/python /tmp/drop_sections.py tests/unit/test_review_fixes.py "C5: recurrent hidden-state reset"
.venv/bin/python /tmp/drop_sections.py tests/unit/test_action_masking.py "ActorCriticNetwork-level tests"
sed -i 's/from colosseum.core.registry import build_network/from colosseum.core.registry import build_model/; s/build_network(cfg)/build_model(cfg)/' tests/unit/test_review_fixes.py
```

What goes and what replaces it: `test_recurrent.py` tested `ActorCriticNetwork` (`is_recurrent`, `initial_hidden`, `act` with `hidden=`, `evaluate_actions(_recurrent)`) and the transitional `build_network` — covered now by `test_cores.py`, `test_model.py`, `test_registry.py`, `test_appo_unroll.py`; the C5 tests in `test_review_fixes.py` (state reset at episode boundaries) are covered by `test_cores.py::test_reset_isolates_post_done_steps` and the T1.6 contract test.

In `tests/unit/test_review_fixes.py` also delete these two lines from the module docstring:

```text
  C5  — recurrent BPTT resets the hidden state at episode boundaries within a
        chunk (post-done timesteps must not depend on pre-done inputs).
```

In `tests/unit/test_action_masking.py` insert this section directly above the `# TrajectoryChunk tests` section header (it replaces the deleted `ActorCriticNetwork`-level tests of `act`/`evaluate_actions` with masks):

```python
# ---------------------------------------------------------------
# PolicyModel-level tests (act / unroll with masks)
# ---------------------------------------------------------------

def _make_model(obs_dim=8, num_actions=4, hidden=32):
    from helpers import make_simple_model

    return make_simple_model(obs_dim=obs_dim, hidden_dim=hidden, num_actions=num_actions)


def test_act_with_mask():
    """act() must only pick legal actions."""
    from colosseum.networks.model import act

    model = _make_model(obs_dim=4, num_actions=5)
    obs = torch.randn(10, 4)
    mask = torch.zeros(10, 5, dtype=torch.bool)
    mask[:, 2] = True  # only action 2 is valid

    out = act(model, obs, None, mask)
    assert (out.actions == 2).all(), f"Expected all actions=2, got {out.actions.tolist()}"
    assert torch.isfinite(out.log_probs).all()
    assert torch.isfinite(out.values).all()


def test_act_without_mask():
    """act() without a mask samples from the full distribution."""
    from colosseum.networks.model import act

    out = act(_make_model(), torch.randn(5, 8), None)
    assert out.actions.shape == (5,)
    assert torch.isfinite(out.log_probs).all()


def test_unroll_with_all_true_mask_matches_unmasked():
    """An all-True mask must not change log-probs or entropy."""
    model = _make_model(obs_dim=4, num_actions=3)
    obs = torch.randn(2, 4, 4)
    dones = torch.zeros(2, 4, dtype=torch.bool)
    actions = torch.randint(0, 3, (8,))
    plain = model.unroll(obs, None, dones)
    masked = model.unroll(obs, None, dones, torch.ones(2, 4, 3, dtype=torch.bool))
    assert torch.allclose(plain.dist.log_prob(actions), masked.dist.log_prob(actions), atol=1e-5)
    assert torch.allclose(plain.dist.entropy(), masked.dist.entropy(), atol=1e-5)
```

- [ ] **Step 9: Check that nothing references the removed API**

Run: `grep -rn "actor_critic\|ActorCritic\|build_network\|make_simple_network\|evaluate_actions\|initial_hidden\|is_recurrent\|lstm_hidden\|network_factor" src tests examples configs scripts`

Expected: no output. (`README.md` and `CLAUDE.md` still describe `recurrent_type` and `actor_critic.py`; they are rewritten in T8.3.)

- [ ] **Step 10: Lint and run the fast suite**

```bash
.venv/bin/ruff check --fix src tests examples scripts && .venv/bin/ruff check src tests examples scripts
.venv/bin/python -m pytest -m "not gpu and not slow" -q
```

Expected: `All checks passed!` (the autofix removes `torch.nn`/`SimpleEncoder`-style imports left unused in `test_review_fixes.py`); all tests pass.

- [ ] **Step 11: Run the slow suite once more**

Run: `.venv/bin/python -m pytest -m slow -q`

Expected: all pass (pipelines, LSTM pipeline, gRPC end-to-end, `torch.compile`).

- [ ] **Step 12: Commit**

```bash
git add -A src tests
git commit -m "refactor: eval, BC and kickstart on PolicyModel; remove ActorCriticNetwork"
```

---

## Contract notes

1. **T1.5 / T1.6 boundary (task content, not signatures).** The index gives T1.5 "APPO on `PolicyModel.unroll`; `TrajectoryChunk.initial_state`" and T1.6 "`RolloutLoop` on `PolicyModel` state + contract test". Removing `lstm_hidden` and switching APPO to `PolicyModel` breaks the worker in the same commit: the worker writes `lstm_hidden` and loads the learner's `state_dict` into an `ActorCriticNetwork`, whose LSTM keys (`recurrent.*`) differ from `ComposedModel`'s (`core.rnn.*`). Keeping the suite green therefore requires switching producer and consumer together. This part does the whole training-path switch (APPO, chunk format, learner, `RolloutLoop` state handling, launcher, distributed) in **T1.5**, and **T1.6** adds the four-core worker → learner contract test, a sensitivity (mutation) check and the reusable `harness.learner_eval`. No contract signature changes; only the split of work between the two IDs moves.

2. **`PolicyModel.unroll` default.** The contract says "default: python loop of step()". The implementation keeps that loop for stateful models and evaluates a stateless model (`state0 is None and not is_stateful`) in a single batched `step` on `[T*B, ...]` (identical results). Joining per-step distributions needs a new classmethod `Distribution.cat(dists)` (implemented for `CategoricalDist`, `DiagGaussianDist`, `CompositeDist`); a user model that relies on the default loop with its own distribution class must implement `cat`. `ComposedModel` overrides `unroll` (encoder and heads batched over `T*B`, `core.unroll` in between), so it never needs `cat`.

3. **`ComposedModel` submodule names.** The contract fixes only the constructor. This part names the submodules `encoder`, `core`, `policy`, `value` (the core of a `NoCore` model has no parameters), so a stateless model's `state_dict` keys equal the old `ActorCriticNetwork` keys and old `.pt` weights of feedforward models still load. T4.2 (`update_normalizers` delegation) and T4.3/T4.4 can rely on these names.

4. **Additions to the contract** (defined in the "Produces" blocks): `APPO.evaluate_chunks(chunks) -> (log_probs [T*B], values [T*B])` (time-major); `RolloutBuffer.initial_state` / `set_initial_state`; `rollout_worker_process(..., model_factories=...)` (renamed from `network_factories`); `colosseum.launcher._create_model`; `evaluate_agents(..., model_factory, model_factories=...)`; `OfflineBCTrainer(model, ...)` with property `model`; `KickstartLoss(teacher_model, ...)`, `compute(student_model, observations)`; `TrajectoryChunk.to(device)` replaces `to_device` (the contract lists `to`); test helpers in `tests/helpers.py` (`CountingEnv`, `make_simple_model`, `make_core`, `CORE_KINDS`, `make_ttt_model`, `example_config`, `REPO_ROOT`, …) and `tests/contract/harness.py` (`make_loop`, `LoopRecorder`, `run_until_chunks`, `weights_payload`, `learner_eval`, …). Later tasks that change `WeightPayload` (T2.2) must update `harness.weights_payload` (it builds a torch `state_dict` today).

5. **`RolloutLoop` initial weight sync** (T0.5) records the payload's `policy_version`; the old worker loaded the initial weights but labelled chunks with version 0 until the first periodic sync. T2.6 (behavior version = first transition) builds on this.

6. **`LoopIO.add_env_steps`** is declared in T0.5 (as in the contract) but not called until T2.5, which decides the batching (spec: about every 0.5 s).

7. **Process-level pipeline tests are `slow` until T2.2.** With one OpenMP thread per process (set by `tests/conftest.py`) they take about 10 s each, but they can fail on the known R6-02 race (torch tensors in `mp.Queue`; a queued item whose sender exited cannot be rebuilt). T2.2 should remove `@pytest.mark.slow` from `tests/integration/test_pipelines.py` (except where a test exceeds ~20 s) and from `tests/integration/test_distributed_e2e.py`, so the fast suite covers the launcher end to end.

8. **Test isolation choices that later parts inherit:** every test runs with `cwd == tmp_path` and `tempfile.tempdir == tmp_path` (autouse fixture), `OMP_NUM_THREADS=1` is exported to spawned children, and `torch.set_num_threads(1)` applies to the pytest process. T2.1 sets torch threads explicitly inside workers/learners, so its tests should assert on the explicit calls, not on the inherited environment. Example configs must be loaded via `helpers.example_config(name)`.

9. **Benchmark measurement source.** `scripts/bench_throughput.py` reads learner metrics by replacing `colosseum.launcher.WandBLogger` at runtime. If T6.3/T6.4 route learner metrics elsewhere, T8.3 must switch the script's `_Recorder` to that source (e.g. `metrics.jsonl`) while keeping the measured quantities and the command line unchanged, so the «после» numbers stay comparable.

10. **Docs not updated in this part.** `README.md` and `CLAUDE.md` still mention `recurrent_type`/`recurrent_hidden_size`/`recurrent_num_layers`, `actor_critic.py`, `evaluate_actions_recurrent` and `pip install -e .`; they are rewritten in T8.3.
