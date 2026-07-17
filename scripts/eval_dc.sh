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
INPUT_DATA="${INPUT_DATA:-data/dc/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv}"
NUM_TRIALS="${NUM_TRIALS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
METRIC_NUM_WORKERS="${METRIC_NUM_WORKERS:-3}"

if [[ ! -f "$INPUT_DATA" ]]; then
  cat >&2 <<EOF
Missing DC input CSV: $INPUT_DATA

Official DC evaluation is pairwise and requires the CSV with question and baseline_answer columns.
Place the official CSV under data/dc/ or pass INPUT_DATA=/path/to/file.
EOF
  exit 2
fi

PYTHON_CMD=(
  "$PYTHON_BIN" -u evaluation/eval_dc.py
  --input-data "$INPUT_DATA"
  --report-dir outputs/dc
  --start-id "${START_ID:-1}"
  --end-id "${END_ID:-0}"
  --limit "${LIMIT:-10}"
  --num-trials "$NUM_TRIALS"
  --num-workers "$NUM_WORKERS"
  --metric-num-workers "$METRIC_NUM_WORKERS"
  --include-images
  --provider "$PROVIDER"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --out-jsonl "${OUT_JSONL:-eval_results/dc/dc_eval_results.jsonl}"
  --out-summary "${OUT_SUMMARY:-eval_results/dc/dc_eval_summary.json}"
)

if [[ -n "$API_URL" ]]; then
  PYTHON_CMD+=(--api-url "$API_URL")
fi

if [[ -n "$API_KEY_ENV" ]]; then
  PYTHON_CMD+=(--api-key-env "$API_KEY_ENV")
fi

"${PYTHON_CMD[@]}"
