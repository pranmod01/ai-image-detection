#!/bin/bash
# Builds this project's Python environment on Insomnia group storage.
# Home is quota-limited (50G), so the env, uv's cache and uv's managed Python
# live under $ZROOT. Override ZROOT to build elsewhere.
set -euo pipefail

ZROOT="${ZROOT:-/insomnia001/depts/zgroup/zgroup_burg/zgroup/users/$USER}"
ENV_DIR="$ZROOT/envs/aidetect"
export UV_CACHE_DIR="$ZROOT/uv_cache"
export UV_PYTHON_INSTALL_DIR="$ZROOT/uv_python"
UV="${UV:-$HOME/.local/bin/uv}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# GPU nodes have driver 575 (CUDA <= 12.9), which rules out cu130 torch builds.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

"$UV" venv --python 3.12 "$ENV_DIR"
export VIRTUAL_ENV="$ENV_DIR"
"$UV" pip install "torch==2.10.0" torchvision --index-url "$TORCH_INDEX"
"$UV" pip install -r "$REPO_DIR/requirements.txt"
echo "Environment ready: source $ENV_DIR/bin/activate"
