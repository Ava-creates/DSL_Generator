#!/usr/bin/env bash
# Submit 5 HuggingFace full pipelines + 5 HuggingFace baseline experiments.
#
# Usage (from repo root):
#   bash scripts/stages/submit_5hf_pipelines_and_baselines.sh

set -euo pipefail

cd /home/avani/projects/aip-lelis/avani/DSL_Generator

PIPELINE_CONFIG="${PIPELINE_CONFIG:-config/experiment_config_hf.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-config/baseline_config_hf.yaml}"
COUNT="${COUNT:-5}"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p scripts/log

echo "=== Submitting ${COUNT} HF pipelines (${PIPELINE_CONFIG}) ==="
PIPELINE_JOBS=()
for i in $(seq 1 "$COUNT"); do
  EXP="experiments/pipeline_hf_${TS}_run${i}_$$"
  export MODEL_TYPE=huggingface
  export EXPERIMENT_DIR="$EXP"
  echo "--- Pipeline ${i}/${COUNT}: ${EXP} ---"
  OUT="$(bash scripts/submit_with_config.sh "$PIPELINE_CONFIG" 2>&1)"
  echo "$OUT"
  JOB_ID="$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)"
  [[ -n "$JOB_ID" ]] && PIPELINE_JOBS+=("$JOB_ID")
  sleep 2
done

echo ""
echo "=== Submitting ${COUNT} HF baselines (${BASELINE_CONFIG}) ==="
BASELINE_JOBS=()
for i in $(seq 1 "$COUNT"); do
  export MODEL_TYPE=huggingface
  export EXPERIMENT_CONFIG="$BASELINE_CONFIG"
  export BASELINE_VARIANT=task_env
  export JOB_PREFIX="baseline_hf_${TS}_run${i}"
  unset EXPERIMENT_DIR
  echo "--- Baseline ${i}/${COUNT} ---"
  OUT="$(bash scripts/stages/submit_baseline.sh 2>&1)"
  echo "$OUT"
  JOB_ID="$(echo "$OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -1 || true)"
  [[ -n "$JOB_ID" ]] && BASELINE_JOBS+=("$JOB_ID")
  sleep 2
done

echo ""
echo "=== Done ==="
echo "Pipeline orchestrator jobs: ${PIPELINE_JOBS[*]:-none}"
echo "Baseline orchestrator jobs: ${BASELINE_JOBS[*]:-none}"
echo "Monitor: squeue -u \$USER"
