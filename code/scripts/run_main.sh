#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${MIRA_DATA_ROOT:?Set the local corruption-data root}"
: "${MIRA_CHECKPOINT_ROOT:?Set the local checkpoint root}"
: "${CUDA_VISIBLE_DEVICES:?Select a GPU before launching evaluation}"
"$PYTHON_BIN" "$REPO_DIR/scripts/check_assets.py" --check-data --scope main
for dataset in c100 in50; do
  for seed in 1 2 3; do
    for method in tent full; do
      "$PYTHON_BIN" "$REPO_DIR/scripts/run_reference.py" --config "$REPO_DIR/configs/${dataset}_${method}_s${seed}.json"
    done
  done
done
