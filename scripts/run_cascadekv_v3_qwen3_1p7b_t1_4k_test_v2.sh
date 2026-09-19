#!/usr/bin/env bash
set -euo pipefail
H=experiments/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.py
uv run python3 "$H" --preflight
for source in narrative report qa; do
  for layer in 0 7 14 21 27; do
    uv run python3 "$H" --capture --source "$source" --layer "$layer"
    uv run python3 "$H" --evaluate --source "$source" --layer "$layer"
  done
done
uv run python3 "$H" --require-complete
uv run python3 "$H" --merge
