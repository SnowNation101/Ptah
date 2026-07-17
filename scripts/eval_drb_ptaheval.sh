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
  "$PYTHON_BIN" -u evaluation/eval_ptaheval.py
  --bench drb
  --report-dir "${REPORT_DIR:-outputs/drb}"
  --html-dir "${HTML_DIR:-outputs/drb}"
  --screenshot-dir "${SCREENSHOT_DIR:-output_screenshots/drb/ptah}"
  --start-id "${START_ID:-1}"
  --end-id "${END_ID:-10}"
  --limit "${LIMIT:-0}"
  --provider "$PROVIDER"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --out-dir "${OUT_DIR:-eval_results/dr_bench/ptaheval/Ptah}"
)

if [[ -n "$API_URL" ]]; then
  CMD+=(--api-url "$API_URL")
fi
if [[ -n "$API_KEY_ENV" ]]; then
  CMD+=(--api-key-env "$API_KEY_ENV")
fi
if [[ "${SKIP_RENDER:-0}" == "1" ]]; then
  CMD+=(--skip-render)
fi
if [[ "${FORCE_RENDER:-0}" == "1" ]]; then
  CMD+=(--force-render)
fi

"${CMD[@]}"
