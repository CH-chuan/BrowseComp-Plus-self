#!/usr/bin/env bash
# Launch vLLM serving a Qwen3 model with tool-call + reasoning parsers configured.
#
# Usage (foreground):
#   ./serve_vllm.sh                       # serves Qwen/Qwen3.5-9B (default)
#   ./serve_vllm.sh Qwen/Qwen3-8B         # serve a different Qwen3 variant
#   MAX_LEN=32768 ./serve_vllm.sh         # override max model length
#
# Background:
#   nohup ./serve_vllm.sh > logs/vllm.log 2>&1 &
#   disown
#
# To stop a running server:
#   pkill -f "vllm serve"
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen3.5-9B}}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.92}"

# Tool-call parser per Qwen3 family (Qwen3.5 / Coder use qwen3_coder; plain Qwen3 uses hermes).
case "$MODEL" in
  *Qwen3.5*|*Qwen3-Coder*) TOOL_PARSER="qwen3_coder" ;;
  *)                       TOOL_PARSER="hermes" ;;
esac

# Default max context per family.
case "$MODEL" in
  *Qwen3.5*) MAX_LEN="${MAX_LEN:-16384}" ;;
  *)         MAX_LEN="${MAX_LEN:-8192}" ;;
esac

echo "=== vLLM launch ==="
echo "  model:          $MODEL"
echo "  port:           $PORT"
echo "  tool-parser:    $TOOL_PARSER"
echo "  reasoning-parser: qwen3"
echo "  max-model-len:  $MAX_LEN"
echo "  gpu-util:       $GPU_UTIL"
echo "==================="

# Restart-safe: kill any existing vLLM server before launching the new one.
if pgrep -f "vllm serve" > /dev/null; then
  echo "killing existing vllm serve process(es)..."
  pkill -f "vllm serve" || true
  # Wait for it to actually die (up to 20s)
  for _ in $(seq 1 20); do
    pgrep -f "vllm serve" > /dev/null || break
    sleep 1
  done
  if pgrep -f "vllm serve" > /dev/null; then
    echo "WARNING: existing vllm serve did not exit cleanly; sending SIGKILL"
    pkill -9 -f "vllm serve" || true
    sleep 2
  fi
fi

exec env \
  LD_LIBRARY_PATH=/home/cche/miniforge3/envs/browsecomp/lib \
  PATH=/home/cche/miniforge3/envs/browsecomp/bin:"$PATH" \
  PYTHONNOUSERSITE=1 \
  HF_HOME=/home/cche/projects/models \
  /home/cche/miniforge3/envs/browsecomp/bin/vllm serve "$MODEL" \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser qwen3 \
    --max-model-len "$MAX_LEN" \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_UTIL"
