#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TARGET_MODEL="${TARGET_MODEL:-Ptah}"
RAW_DATA_DIR="${RAW_DATA_DIR:-outputs/drb}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/dr_bench/race/${TARGET_MODEL}}"
QUERY_DATA_PATH="${QUERY_DATA_PATH:-data/drb/query.jsonl}"
CRITERIA_PATH="${CRITERIA_PATH:-data/drb/criteria.jsonl}"
REFERENCE_PATH="${REFERENCE_PATH:-data/drb/reference.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROVIDER="${PROVIDER:-uniapi}"
MODEL="${MODEL:-qwen3-vl-235b-a22b-instruct}"
BASE_URL="${BASE_URL:-https://hk.uniapi.io/v1}"
API_URL="${API_URL:-}"
API_KEY_ENV="${API_KEY_ENV:-}"

CMD=(
  "$PYTHON_BIN" -u evaluation/eval_dr_bench.py "$TARGET_MODEL"
  --raw_data_dir "$RAW_DATA_DIR"
  --max_workers "${MAX_WORKERS:-1}"
  --query_file "$QUERY_DATA_PATH"
  --criteria_file "$CRITERIA_PATH"
  --reference_file "$REFERENCE_PATH"
  --output_dir "$OUTPUT_DIR"
  --provider "$PROVIDER"
  --judge_model "$MODEL"
  --base_url "$BASE_URL"
  --skip_cleaning
  --force
)

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ "${ONLY_ZH:-0}" == "1" ]]; then
  CMD+=(--only_zh)
fi
if [[ "${ONLY_EN:-0}" == "1" ]]; then
  CMD+=(--only_en)
fi
if [[ "${INCLUDE_IMAGES:-0}" == "1" ]]; then
  CMD+=(--include_images)
fi
if [[ -n "$API_URL" ]]; then
  CMD+=(--api_url "$API_URL")
fi
if [[ -n "$API_KEY_ENV" ]]; then
  CMD+=(--api_key_env "$API_KEY_ENV")
fi

"${CMD[@]}"
