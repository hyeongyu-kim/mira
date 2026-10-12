#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${MIRA_DATA_ROOT:?Set the local corruption-data root}"
: "${MIRA_CHECKPOINT_ROOT:?Set the local checkpoint root}"
: "${CUDA_VISIBLE_DEVICES:?Select a GPU before launching evaluation}"
"$PYTHON_BIN" "$REPO_DIR/scripts/run_ablations.py" "$@"
