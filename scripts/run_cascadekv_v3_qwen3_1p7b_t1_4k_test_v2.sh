#!/usr/bin/env bash
set -euo pipefail
# Protocol preparation only: this driver deliberately exposes no execution phase.
uv run python3 experiments/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.py --preflight
uv run python3 experiments/cascadekv_v3_qwen3_1p7b_t1_4k_test_v2.py --status
