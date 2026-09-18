#!/usr/bin/env bash
# Preparation-only wrapper: no model/data driver is enabled.
set -euo pipefail
prog=experiments/cascadekv_v3_qwen3_1p7b_wide_vaware_dev.py
uv run python3 "$prog" --preflight
uv run python3 "$prog" --status
