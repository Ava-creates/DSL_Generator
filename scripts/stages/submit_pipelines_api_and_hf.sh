#!/usr/bin/env bash
# Submit two full DSL pipelines: OpenAI-compatible API vs local HuggingFace.
# Uses scripts/submit_with_config.sh (stage_get_cfg → chain_next_stage).
#
# Usage (from repo root):
#   export OPENAI_COMPAT_KEY_FILE="$PWD/key.txt"
#   bash scripts/stages/submit_pipelines_api_and_hf.sh [config.yaml]
#
# First arg defaults to config/experiment_config.yaml.
# Override experiment roots with PIPELINE_EXPERIMENT_DIR_API / PIPELINE_EXPERIMENT_DIR_HF if needed.

set -euo pipefail

cd /home/avani/projects/aip-lelis/avani/DSL_Generator

CONFIG_FILE="${1:-config/experiment_config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config not found: $CONFIG_FILE" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"

submit_one() {
  local model="$1"
  local label="$2"
  local expdir="$3"

  export MODEL_TYPE="$model"
  export EXPERIMENT_DIR="$expdir"
  export EXPERIMENT_CONFIG="$CONFIG_FILE"

  if [ "$model" = "openai_compat" ]; then
    if [ -z "${OPENAI_COMPAT_KEY_FILE:-}" ] && [ -z "${OPENAI_COMPAT_API_KEY:-}" ]; then
      echo "ERROR: set OPENAI_COMPAT_KEY_FILE or OPENAI_COMPAT_API_KEY for API pipeline run." >&2
      exit 1
    fi
  else
    unset OPENAI_COMPAT_KEY_FILE 2>/dev/null || true
  fi

  echo "=== Submitting pipeline MODEL_TYPE=$model EXPERIMENT_DIR=$expdir ==="
  bash scripts/submit_with_config.sh "$CONFIG_FILE"
}

API_DIR="${PIPELINE_EXPERIMENT_DIR_API:-experiments/pipeline_api_${TS}_$$}"
HF_DIR="${PIPELINE_EXPERIMENT_DIR_HF:-experiments/pipeline_hf_${TS}_$$}"

submit_one openai_compat api "$API_DIR"
submit_one huggingface hf "$HF_DIR"

echo "Done. Logs under scripts/log/<experiment_dir_basename>/stage_get_cfg_*.out"
