#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1/}"
MODEL_NAME="${MODEL_NAME:-models/Qwen3-32B}"
REVIEWER_BASE_URL="${REVIEWER_BASE_URL:-$BASE_URL}"
REVIEWER_MODEL_NAME="${REVIEWER_MODEL_NAME:-$MODEL_NAME}"
SECTION_WORKERS="${SECTION_WORKERS:-4}"
CURRENT_DATE="${CURRENT_DATE:-$(date +%F)}"
QUESTION="${QUESTION:-帮我调研切尔西足球队的历史。}"

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/custom_${TIMESTAMP}.log"

echo "Starting custom Ptah report. Logging to ${LOG_FILE}"
echo "Question: ${QUESTION}"

"$PYTHON_BIN" -u main.py \
  --base_url "$BASE_URL" \
  --model_name "$MODEL_NAME" \
  --reviewer_base_url "$REVIEWER_BASE_URL" \
  --reviewer_model_name "$REVIEWER_MODEL_NAME" \
  --task custom \
  --section_workers "$SECTION_WORKERS" \
  --current_date "$CURRENT_DATE" \
  --question "$QUESTION" 2>&1 | tee "$LOG_FILE"
