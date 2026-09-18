#!/usr/bin/env bash
set -euo pipefail
# Pre-capture-only runner.  It intentionally cannot capture or evaluate.
ROOT="${RESULTS_DIR:-results}"
uv run python3 experiments/cascadekv_v3_qwen3_1p7b_4k_test.py --preflight
uv run python3 experiments/cascadekv_v3_qwen3_1p7b_4k_test.py --status --cache-dir "$ROOT/cascadekv_v3_qwen3_1p7b_4k_test_cache" --shard-dir "$ROOT/cascadekv_v3_qwen3_1p7b_4k_test_shards"
