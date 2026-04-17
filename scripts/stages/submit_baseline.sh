#!/bin/bash
set -euo pipefail

cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Track whether caller explicitly provided EXPERIMENT_DIR before config loading.
CALLER_EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"

# Optionally load YAML config if provided or default baseline config exists.
if [ -n "${EXPERIMENT_CONFIG:-}" ] || [ -f "config/baseline_config.yaml" ]; then
  source scripts/load_config.sh
fi

BASELINE_VARIANT="${BASELINE_VARIANT:-testcase}"
PHASE2_ONLY="${PHASE2_ONLY:-false}"

# By default, create a unique baseline experiment directory per submission
# to avoid multiple jobs writing into the same folder.
# Disabled automatically for phase2_only resumes, or by setting BASELINE_UNIQUE_DIR=false.
BASELINE_UNIQUE_DIR="${BASELINE_UNIQUE_DIR:-true}"
if [ "$PHASE2_ONLY" != "true" ] && [ "$PHASE2_ONLY" != "1" ] && [ "$BASELINE_UNIQUE_DIR" != "false" ] && [ "$BASELINE_UNIQUE_DIR" != "0" ]; then
  if [ -z "$CALLER_EXPERIMENT_DIR" ]; then
    BASE_DIR="${EXPERIMENT_DIR:-experiments/baseline}"
    SAFE_VARIANT="${BASELINE_VARIANT//[^a-zA-Z0-9_-]/_}"
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    EXPERIMENT_DIR="${BASE_DIR}_${SAFE_VARIANT}_${TIMESTAMP}_$$"
    export EXPERIMENT_DIR
  fi
fi

# Allow explicit override when needed.
if [ -n "${BASELINE_WALLTIME:-}" ]; then
  WALLTIME="$BASELINE_WALLTIME"
else
  if [ "$BASELINE_VARIANT" = "task_env" ] || [ "$PHASE2_ONLY" = "true" ] || [ "$PHASE2_ONLY" = "1" ]; then
    WALLTIME="04:00:00"
  else
    WALLTIME="12:00:00"
  fi
fi

echo "[submit_baseline] BASELINE_VARIANT=$BASELINE_VARIANT PHASE2_ONLY=$PHASE2_ONLY"
echo "[submit_baseline] EXPERIMENT_DIR=${EXPERIMENT_DIR:-<stage-default>}"
echo "[submit_baseline] Submitting with --time=$WALLTIME"

if [ -z "${EXPERIMENT_DIR:-}" ]; then
  EXPERIMENT_DIR="experiments/baseline_$(date +%Y%m%d_%H%M%S)_$$"
  export EXPERIMENT_DIR
fi

BASELINE_LOG_DIR="${BASELINE_LOG_DIR:-scripts/log/${EXPERIMENT_DIR}}"
export BASELINE_LOG_DIR
mkdir -p "$BASELINE_LOG_DIR"

echo "[submit_baseline] BASELINE_LOG_DIR=$BASELINE_LOG_DIR"

# When using the OpenAI-compatible API the orchestrator job itself doesn't need
# GPUs (model inference happens remotely).  Override the script's #SBATCH
# defaults so the job lands on a CPU-only partition faster.
SBATCH_GPU_ARGS=""
if [ "${MODEL_TYPE:-huggingface}" = "openai_compat" ]; then
  SBATCH_GPU_ARGS="--gres=gpu:0 --mem=32G --cpus-per-task=4"
  echo "[submit_baseline] API mode detected: submitting orchestrator without GPUs"
fi

# shellcheck disable=SC2086
sbatch \
  --time "$WALLTIME" \
  --output "$BASELINE_LOG_DIR/stage_baseline_%j.out" \
  --error "$BASELINE_LOG_DIR/stage_baseline_%j.err" \
  $SBATCH_GPU_ARGS \
  scripts/stages/stage_baseline.slurm
