#!/usr/bin/env bash
set -euo pipefail
ROOT="${RESULTS_DIR:-results}"
HARNESS="experiments/cascadekv_v3_qwen3_1p7b_4k_test.py"
CACHE="$ROOT/cascadekv_v3_qwen3_1p7b_4k_test_cache"
SHARDS="$ROOT/cascadekv_v3_qwen3_1p7b_4k_test_shards"
OUTPUT="$ROOT/cascadekv_v3_qwen3_1p7b_4k_test.json"
uv run python3 "$HARNESS" --preflight
for sequence in narrative report qa; do
  for layer in 0 7 14 21 27; do
    # Each phase validates internally.  We invoke it unconditionally so a
    # corrupt or obsolete existing path is never treated as resumable merely
    # because it exists; valid artifacts return without work.
    uv run python3 "$HARNESS" --capture --sequence "$sequence" --layer "$layer" --cache-dir "$CACHE" --shard-dir "$SHARDS"
    uv run python3 "$HARNESS" --evaluate --sequence "$sequence" --layer "$layer" --cache-dir "$CACHE" --shard-dir "$SHARDS"
  done
done
uv run python3 "$HARNESS" --merge --shard-dir "$SHARDS" --output "$OUTPUT"
