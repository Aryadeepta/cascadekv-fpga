#!/usr/bin/env bash
set -euo pipefail

# Future production driver. It is deliberately never invoked by this task.
runner=(uv run python3 experiments/cascadekv_v3_qwen3_1p7b_t1_4k_test.py)
"${runner[@]}" --preflight
for source in narrative report qa; do
  for layer in 0 7 14 21 27; do
    "${runner[@]}" --capture --source "$source" --layer "$layer"
    "${runner[@]}" --evaluate --source "$source" --layer "$layer"
  done
done
"${runner[@]}" --require-complete
"${runner[@]}" --merge
