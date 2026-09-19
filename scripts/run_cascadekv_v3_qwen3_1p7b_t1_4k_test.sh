#!/usr/bin/env bash
set -euo pipefail

# Preparation-only protocol: this runner can only validate frozen provenance
# and report the future 15-cache/15-shard empty state.
mode="${1:---preflight}"

case "$mode" in
  --preflight|--status) exec uv run python3 experiments/cascadekv_v3_qwen3_1p7b_t1_4k_test.py "$mode" ;;
  *) echo "Only --preflight and --status are enabled during protocol preparation." >&2; exit 2 ;;
esac
