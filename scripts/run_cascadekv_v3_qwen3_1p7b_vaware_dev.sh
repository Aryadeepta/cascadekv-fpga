#!/usr/bin/env bash
set -euo pipefail
# Protocol preparation only: intentionally does not capture, evaluate, merge, or optimize.
uv run python3 experiments/cascadekv_v3_qwen3_1p7b_vaware_dev.py --preflight
uv run python3 experiments/cascadekv_v3_qwen3_1p7b_vaware_dev.py --status
