#!/usr/bin/env bash
# Run the sampling-tree client against a Qwen3 vLLM server.
#
# Defaults are tuned for Qwen3.5-9B with thinking disabled (per Qwen non-thinking spec).
# All sampling_tree_client.py flags can be passed through as additional args.
#
# Usage:
#   ./run_sampling_tree.sh                                          # smoke: depth 2, --limit 1, curated50
#   ./run_sampling_tree.sh --depth 4 --limit 5                      # bigger smoke
#   ./run_sampling_tree.sh --dataset data/browsecomp_plus_decrypted.jsonl  # full sweep (no --limit)
#   ./run_sampling_tree.sh --run-name smoke1 ...                    # resume / namespace
#
# Override env:
#   OUTPUT_DIR=output-test/my_run ./run_sampling_tree.sh
#   MODEL=Qwen/Qwen3-8B ./run_sampling_tree.sh                      # different model
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output-test/sampling_tree}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MODEL_URL="${MODEL_URL:-http://localhost:8000/v1}"
DATASET="${DATASET:-topics-qrels/curated50.jsonl}"
RETRIEVERS_CONFIG="${RETRIEVERS_CONFIG:-configs/retrievers.json}"
MAX_TOKENS="${MAX_TOKENS:-6000}"

mkdir -p logs

exec env \
  JAVA_HOME=/home/cche/miniforge3/envs/browsecomp/lib/jvm \
  JVM_PATH=/home/cche/miniforge3/envs/browsecomp/lib/jvm/lib/server/libjvm.so \
  PYTHONNOUSERSITE=1 \
  LD_LIBRARY_PATH=/home/cche/miniforge3/envs/browsecomp/lib \
  PATH=/home/cche/miniforge3/envs/browsecomp/bin:"$PATH" \
  HF_HOME=/home/cche/projects/models \
  /home/cche/miniforge3/envs/browsecomp/bin/python search_agent/sampling_tree_client.py \
    --dataset "$DATASET" \
    --gold-source gold \
    --output-dir "$OUTPUT_DIR" \
    --retrievers-config "$RETRIEVERS_CONFIG" \
    --model "$MODEL" \
    --model-url "$MODEL_URL" \
    --max-tokens "$MAX_TOKENS" \
    "$@"
