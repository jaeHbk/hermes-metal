#!/usr/bin/env bash
# bench/setup.sh
#
# Provisions a SEPARATE Python venv (.venv-mlx) for the MLX side of the
# benchmark. Kept isolated from the daemon's .venv so MLX's PyTorch /
# transformers chain cannot drift the runtime stack of the LaunchAgents.
#
# Idempotent: rerunnable. Safe on Apple Silicon arm64 only.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
MLX_VENV="$BENCH_DIR/.venv-mlx"
PY="${PYTHON:-python3}"

# ---- arch guard --------------------------------------------------------------

if [ "$(uname -m)" != "arm64" ]; then
    echo "ERROR: bench/setup.sh requires Apple Silicon (arm64). Detected: $(uname -m)" >&2
    exit 1
fi

# ---- mlx venv ----------------------------------------------------------------

echo "==> bench/setup.sh"
echo "    repo root:  $REPO_ROOT"
echo "    mlx venv:   $MLX_VENV"

if [ ! -d "$MLX_VENV" ]; then
    echo "    creating MLX venv"
    "$PY" -m venv "$MLX_VENV"
fi

"$MLX_VENV/bin/pip" install --upgrade pip >/dev/null

# Pin the MLX side. mlx-lm pulls in mlx + transformers + tokenizers as deps.
# Versions verified 2026-06-03: mlx-lm 0.31.3 is the current PyPI release.
# We don't pin numpy: mlx-lm declares only `numpy` (any version) and pinning
# numpy<2.1 forces a source build on Python 3.14 (no cp314 wheels for 1.x).
"$MLX_VENV/bin/pip" install --prefer-binary --upgrade \
    "mlx-lm>=0.31,<0.32" \
    "psutil>=5.9" \
    "datasets>=2.18"

# ---- daemon-venv extras for the llama.cpp side -------------------------------
# The daemon's existing .venv already has httpx + numpy. We need psutil + datasets
# on that side too for parity with the MLX harness.

DAEMON_VENV="$REPO_ROOT/.venv"
if [ -x "$DAEMON_VENV/bin/pip" ]; then
    echo "    extending daemon venv with bench deps"
    "$DAEMON_VENV/bin/pip" install --prefer-binary --upgrade \
        "psutil>=5.9" \
        "datasets>=2.18" >/dev/null
else
    echo "    WARNING: daemon venv $DAEMON_VENV not found."
    echo "             Run 'make setup-venv' from repo root, then re-run bench/setup.sh."
fi

# ---- sanity check ------------------------------------------------------------

"$MLX_VENV/bin/python" - <<'PY'
import platform, sys
import mlx.core as mx
from mlx_lm import __version__ as v
assert platform.machine() == "arm64", platform.machine()
print(f"    mlx-lm:     {v}")
print(f"    mlx metal:  {mx.metal.is_available()}")
print(f"    python:     {sys.version.split()[0]} on {platform.machine()}")
PY

echo "    done. next: bench/bench_throughput.py --backend both"
