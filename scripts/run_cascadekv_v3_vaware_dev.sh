#!/usr/bin/env bash
# Resumable v3 development run.  --status is model-free; capture is explicit.
set -euo pipefail
ROOT="${RESULTS_DIR:-results}"
CACHE="$ROOT/cascadekv_v3_vaware_dev_cache"
SHARDS="$ROOT/cascadekv_v3_vaware_dev_shards"
LOGS="$ROOT/cascadekv_v3_vaware_dev_logs"
mkdir -p "$LOGS"
uv run python3 experiments/cascadekv_v3_vaware_dev.py --preflight
uv run python3 experiments/cascadekv_v3_vaware_dev.py --status --cache-dir "$CACHE" --shard-dir "$SHARDS" --status-output "$LOGS/status.json"
for seq in narrative_calibration narrative_validation report_calibration report_validation qa_calibration qa_validation; do
  for layer in 0 7 14 21 27; do
    cache="$CACHE/${seq}_L4096_layer${layer}_qkv.pt"
    shard="$SHARDS/${seq}_L4096_layer${layer}.json"
    if ! uv run python3 experiments/cascadekv_v3_vaware_dev.py --validate-cache --sequence "$seq" --layer "$layer" --cache-dir "$CACHE" >"$LOGS/${seq}_L4096_layer${layer}.cache-validation.log" 2>&1; then
      echo "[capture] $seq layer $layer: cache missing or invalid; regenerating"
      uv run python3 experiments/cascadekv_v3_vaware_dev.py --capture --sequence "$seq" --layer "$layer" --cache-dir "$CACHE" >"$LOGS/${seq}_L4096_layer${layer}.capture.log" 2>&1
      uv run python3 experiments/cascadekv_v3_vaware_dev.py --validate-cache --sequence "$seq" --layer "$layer" --cache-dir "$CACHE" >"$LOGS/${seq}_L4096_layer${layer}.cache-validation.log" 2>&1
    else
      echo "[capture] $seq layer $layer: valid cache; skipping"
    fi
    if ! uv run python3 experiments/cascadekv_v3_vaware_dev.py --validate-shard --sequence "$seq" --layer "$layer" --shard-dir "$SHARDS" >"$LOGS/${seq}_L4096_layer${layer}.shard-validation.log" 2>&1; then
      echo "[evaluate] $seq layer $layer: shard missing or invalid; regenerating"
      uv run python3 experiments/cascadekv_v3_vaware_dev.py --evaluate --sequence "$seq" --layer "$layer" --cache-dir "$CACHE" --shard-dir "$SHARDS" >"$LOGS/${seq}_L4096_layer${layer}.evaluate.log" 2>&1
      uv run python3 experiments/cascadekv_v3_vaware_dev.py --validate-shard --sequence "$seq" --layer "$layer" --shard-dir "$SHARDS" >"$LOGS/${seq}_L4096_layer${layer}.shard-validation.log" 2>&1
    else
      echo "[evaluate] $seq layer $layer: valid shard; skipping"
    fi
  done
done
uv run python3 experiments/cascadekv_v3_vaware_dev.py --merge --shard-dir "$SHARDS" --output "$ROOT/cascadekv_v3_vaware_dev.json"
