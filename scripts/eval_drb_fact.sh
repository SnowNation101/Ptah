#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
PROVIDER="${PROVIDER:-uniapi}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"
BASE_URL="${BASE_URL:-https://hk.uniapi.io/v1}"
API_URL="${API_URL:-}"
API_KEY_ENV="${API_KEY_ENV:-}"

CMD=(
  "$PYTHON_BIN" -u evaluation/eval_drb_fact.py
  --query-file "${QUERY_DATA_PATH:-data/drb/query.jsonl}"
  --report-dir "${REPORT_DIR:-outputs/drb}"
  --start-id "${START_ID:-1}"
  --end-id "${END_ID:-0}"
  --limit "${LIMIT:-0}"
  --provider "$PROVIDER"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --max-workers "${MAX_WORKERS:-4}"
  --out-jsonl "${OUT_JSONL:-eval_results/dr_bench/fact/Ptah/raw_results.jsonl}"
  --out-summary "${OUT_SUMMARY:-eval_results/dr_bench/fact/Ptah/fact_summary.json}"
)

if [[ -n "$API_URL" ]]; then
  CMD+=(--api-url "$API_URL")
fi
if [[ -n "$API_KEY_ENV" ]]; then
  CMD+=(--api-key-env "$API_KEY_ENV")
fi

"${CMD[@]}"
