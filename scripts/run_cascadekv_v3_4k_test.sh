#!/usr/bin/env bash
set -euo pipefail
# This is the only command intended to start/resume the already-frozen test.
ROOT="${RESULTS_DIR:-results}"
CACHE="$ROOT/cascadekv_v3_4k_test_cache"; SHARDS="$ROOT/cascadekv_v3_4k_test_shards"; LOGS="$ROOT/cascadekv_v3_4k_test_logs"
mkdir -p "$LOGS"
run() { uv run python3 experiments/cascadekv_v3_4k_test.py "$@" >>"$LOGS/driver.log" 2>&1; }
valid() { uv run python3 experiments/cascadekv_v3_4k_test.py --status --cache-dir "$CACHE" --shard-dir "$SHARDS" | python3 -c 'import json,sys;d=json.load(sys.stdin);s,l,k=sys.argv[1:];print(next(x for x in d["pairs"] if x["sequence"]==s and x["layer"]==int(l))[k]["status"])' "$1" "$2" "$3"; }
run --preflight
for s in narrative report qa; do for l in 0 7 14 21 27; do
  [[ "$(valid "$s" "$l" cache)" == valid ]] || run --capture --sequence "$s" --layer "$l" --cache-dir "$CACHE"
  [[ "$(valid "$s" "$l" shard)" == valid ]] || run --evaluate --sequence "$s" --layer "$l" --cache-dir "$CACHE" --shard-dir "$SHARDS"
done; done
run --merge --shard-dir "$SHARDS" --output "$ROOT/cascadekv_v3_4k_test.json"
