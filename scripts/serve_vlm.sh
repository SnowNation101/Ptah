#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP_SIZE="${TP_SIZE:-2}"
VLLM_BIN="${VLLM_BIN:-vllm}"

mkdir -p logs

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --limit-mm-per-prompt.video 0 \
  --reasoning-parser qwen3
