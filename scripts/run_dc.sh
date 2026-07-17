#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs
LOG_FILE="logs/dc_${TIMESTAMP}.log"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1/}"
MODEL_NAME="${MODEL_NAME:-models/Qwen3-32B}"
REVIEWER_BASE_URL="${REVIEWER_BASE_URL:-$BASE_URL}"
REVIEWER_MODEL_NAME="${REVIEWER_MODEL_NAME:-$MODEL_NAME}"
SECTION_WORKERS="${SECTION_WORKERS:-4}"
CURRENT_DATE="${CURRENT_DATE:-$(date +%F)}"

echo "Starting Deep Consult task... Logging to ${LOG_FILE}"


"$PYTHON_BIN" -u main.py \
    --base_url "$BASE_URL" \
    --model_name "$MODEL_NAME" \
    --reviewer_base_url "$REVIEWER_BASE_URL" \
    --reviewer_model_name "$REVIEWER_MODEL_NAME" \
    --task "dc" \
    --section_workers "$SECTION_WORKERS" \
    --current_date "$CURRENT_DATE" 2>&1 | tee "$LOG_FILE"


if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Task completed successfully."
else
    echo "Task failed. Please check $LOG_FILE for details."
fi
