#!/usr/bin/env bash
# Full sweep: all 50 queries in topics-qrels/curated50.jsonl at depth 10.
#
# Self-daemonizing — runs in the background; safe to close your session after
# kickoff. Resume by passing the SAME RUN_NAME env var on a later invocation.
#
# Defaults:
#   - run_name:       curated50_d10_<utc_timestamp>  (fresh each invocation)
#   - output_dir:     output-runs/$RUN_NAME/
#   - log:            logs/full_$RUN_NAME.log
#   - depth:          10
#   - n_samples:      10
#   - sample_workers: 10
#
# Usage:
#   ./run_full_curated50.sh                                # fresh run
#   RUN_NAME=curated50_d10_20260511T031500Z ./run_full_curated50.sh  # resume
#   DEPTH=6 ./run_full_curated50.sh                        # override depth
#
# Monitoring:
#   tail -f logs/full_<RUN_NAME>.log
#   ls output-runs/<RUN_NAME>/
#   wc -l output-runs/<RUN_NAME>/*/summary.json     # count completed qids
#
# Stop:
#   pkill -f sampling_tree_client.py
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-output-runs}"
DEPTH="${DEPTH:-10}"
N_SAMPLES="${N_SAMPLES:-10}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-10}"
DATASET="${DATASET:-topics-qrels/curated50.jsonl}"

# Step 1: top-level invocation. Build run name + log path, self-daemonize.
if [ "${_RUN_DAEMONIZED:-}" != "1" ]; then
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  RUN_PREFIX="${RUN_PREFIX:-curated50_d${DEPTH}_n${N_SAMPLES}}"
  RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_$TS}"
  LOG_FILE="logs/full_${RUN_NAME}.log"
  mkdir -p logs

  echo "=== full curated50 sweep (background) ==="
  echo "  run_name:       $RUN_NAME"
  echo "  output dir:     $OUTPUT_ROOT/$RUN_NAME/"
  echo "  log:            $LOG_FILE"
  echo "  depth:          $DEPTH"
  echo "  n_samples:      $N_SAMPLES"
  echo "  sample_workers: $SAMPLE_WORKERS"
  echo "  dataset:        $DATASET"
  echo "=========================================="

  _RUN_DAEMONIZED=1 RUN_NAME="$RUN_NAME" \
    OUTPUT_ROOT="$OUTPUT_ROOT" DEPTH="$DEPTH" \
    N_SAMPLES="$N_SAMPLES" SAMPLE_WORKERS="$SAMPLE_WORKERS" \
    DATASET="$DATASET" \
    nohup "$0" "$@" > "$LOG_FILE" 2>&1 &
  disown
  PID=$!

  echo "  pid:            $PID"
  echo ""
  echo "  monitor:        tail -f $LOG_FILE"
  echo "  progress:       ls $OUTPUT_ROOT/$RUN_NAME/ | wc -l    # qids started"
  echo "                  find $OUTPUT_ROOT/$RUN_NAME -name summary.json | wc -l   # qids done"
  echo "  resume later:   RUN_NAME=$RUN_NAME $0"
  echo "  stop:           pkill -f sampling_tree_client.py"
  exit 0
fi

# Step 2: daemon. Run the sweep.
echo "=== daemon start: $(date -u) ==="
echo "  run_name:       $RUN_NAME"
echo "  output_root:    $OUTPUT_ROOT"
echo "  dataset:        $DATASET"
echo "  depth:          $DEPTH"
echo "  n_samples:      $N_SAMPLES"
echo "  sample_workers: $SAMPLE_WORKERS"
echo

exec ./run_sampling_tree.sh \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_ROOT" \
  --run-name "$RUN_NAME" \
  --depth "$DEPTH" \
  --n-samples "$N_SAMPLES" \
  --sample-workers "$SAMPLE_WORKERS" \
  "$@"
