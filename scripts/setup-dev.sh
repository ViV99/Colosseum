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
