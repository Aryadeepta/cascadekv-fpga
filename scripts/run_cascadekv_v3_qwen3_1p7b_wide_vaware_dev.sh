#!/usr/bin/env bash
set -euo pipefail
prog=experiments/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.py
cache=results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_cache
shards=results/cascadekv_v3_qwen3_1p7b_wide_vaware_dev_shards
uv run python3 "$prog" --preflight
for source in narrative_calibration narrative_validation report_calibration report_validation qa_calibration qa_validation; do
  for layer in 0 7 14 21 27; do
    uv run python3 "$prog" --capture --source "$source" --layer "$layer" --cache-dir "$cache"
    uv run python3 "$prog" --evaluate --source "$source" --layer "$layer" --cache-dir "$cache" --shard-dir "$shards"
  done
done
uv run python3 "$prog" --status --cache-dir "$cache" --shard-dir "$shards"
uv run python3 "$prog" --require-complete --cache-dir "$cache" --shard-dir "$shards"
uv run python3 "$prog" --merge --cache-dir "$cache" --shard-dir "$shards"
