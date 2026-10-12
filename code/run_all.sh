#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-main}" in
  main)
    bash "$REPO_DIR/scripts/run_main.sh"
    bash "$REPO_DIR/scripts/run_vit.sh"
    ;;
  vit)
    shift
    bash "$REPO_DIR/scripts/run_vit.sh" "$@"
    ;;
  ablations)
    bash "$REPO_DIR/scripts/run_ablations.sh"
    ;;
  all)
    bash "$REPO_DIR/scripts/run_main.sh"
    bash "$REPO_DIR/scripts/run_vit.sh"
    bash "$REPO_DIR/scripts/run_ablations.sh"
    ;;
  *)
    echo 'Usage: bash run_all.sh [main|vit|ablations|all]; vit accepts --profile and --seeds' >&2
    exit 2
    ;;
esac
