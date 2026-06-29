#!/bin/bash
# Helper script to submit pipeline with config file
# Usage: ./scripts/submit_with_config.sh [config_file.yaml]
#
# API mode: set model_type: openai_compat in the YAML (or export MODEL_TYPE=openai_compat before running).
#   Uses OPENAI_COMPAT_API_KEY if set; otherwise defaults OPENAI_COMPAT_KEY_FILE to <repo>/key.txt.
#
# Run from anywhere; the script cds to the repo root.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Get config file path
CONFIG_FILE="${1:-config/experiment_config.yaml}"
if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_FILE="${REPO_ROOT}/${CONFIG_FILE}"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Usage: $0 [config_file.yaml]"
    echo ""
    echo "Example:"
    echo "  $0 config/experiment_config.yaml"
    echo ""
    echo "Or set EXPERIMENT_CONFIG environment variable:"
    echo "  export EXPERIMENT_CONFIG=my_config.yaml"
    echo "  $0"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo ""

# Load config and export to environment
export EXPERIMENT_CONFIG="$CONFIG_FILE"
source scripts/load_config.sh

# API: default key file to repo key.txt when no API key in environment
if [ "${MODEL_TYPE:-huggingface}" = "openai_compat" ]; then
    if [ -z "${OPENAI_COMPAT_API_KEY:-}" ]; then
        export OPENAI_COMPAT_KEY_FILE="${OPENAI_COMPAT_KEY_FILE:-${REPO_ROOT}/key.txt}"
        if [ ! -f "${OPENAI_COMPAT_KEY_FILE}" ]; then
            echo "Warning: key file not found at ${OPENAI_COMPAT_KEY_FILE} (set OPENAI_COMPAT_API_KEY or create the file)." >&2
        fi
    fi
fi

# Ensure each submission has an isolated experiment directory unless explicitly provided
if [ -z "${EXPERIMENT_DIR:-}" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_DIR="experiments/experiment_${timestamp}_$RANDOM"
    export EXPERIMENT_DIR
    echo "Auto-generated EXPERIMENT_DIR: $EXPERIMENT_DIR"
    echo ""
fi

# Verify required variables are set
if [ -z "${TASKS:-}" ]; then
    echo "Error: TASKS not set in config file or environment"
    exit 1
fi

if [ -z "${SPEC_FILE:-}" ]; then
    echo "Error: SPEC_FILE not set in config file or environment"
    exit 1
fi

echo "Configuration loaded:"
echo "  EXPERIMENT_DIR: ${EXPERIMENT_DIR:-<auto-generate>}"
echo "  SPEC_FILE: $SPEC_FILE"
echo "  TASKS: $TASKS"
echo "  MODEL_TYPE: ${MODEL_TYPE:-huggingface}"
echo "  TOTAL_SAMPLES: ${TOTAL_SAMPLES:-1000}"
echo "  MAX_DSL_EVOLUTIONS: ${MAX_DSL_EVOLUTIONS:-2}"
echo "  MAX_FUNCTION_EVOLUTIONS: ${MAX_FUNCTION_EVOLUTIONS:-1}"
if [ "${MODEL_TYPE:-huggingface}" = "openai_compat" ]; then
    if [ -n "${OPENAI_COMPAT_API_KEY:-}" ]; then
        echo "  API key: OPENAI_COMPAT_API_KEY (set)"
    else
        echo "  API key file: ${OPENAI_COMPAT_KEY_FILE:-}"
    fi
fi
echo ""

# API mode: wait for Vulcan model cold start before flooding the endpoint
if [ "${MODEL_TYPE:-huggingface}" = "openai_compat" ]; then
    echo "Waiting for Vulcan API model cold start..."
    python3 -m src.utils.openai_compat_cold_start
    echo ""
fi

# Submit the first job
echo "Submitting pipeline job..."
# Use experiment-specific log folder when EXPERIMENT_DIR is set
LOG_DIR="scripts/log"
if [ -n "${EXPERIMENT_DIR:-}" ]; then
    LOG_DIR="scripts/log/$(basename "$EXPERIMENT_DIR")"
fi
ABS_LOG_DIR="${REPO_ROOT}/${LOG_DIR#./}"
mkdir -p "$ABS_LOG_DIR"
EXPORT_VARS="ALL,EXPERIMENT_CONFIG=${CONFIG_FILE},DSL_GENERATOR_ROOT=${REPO_ROOT},MODEL_TYPE=${MODEL_TYPE:-huggingface}"
if [ -n "${EXPERIMENT_DIR:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},EXPERIMENT_DIR=${EXPERIMENT_DIR}"
fi
if [ -n "${OPENAI_COMPAT_KEY_FILE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},OPENAI_COMPAT_KEY_FILE=${OPENAI_COMPAT_KEY_FILE}"
fi
SBATCH_CMD=(
    sbatch
    --chdir="${REPO_ROOT}"
    --export="${EXPORT_VARS}"
    --output="${ABS_LOG_DIR}/stage_get_cfg_%j.out"
    --error="${ABS_LOG_DIR}/stage_get_cfg_%j.err"
)

if [ "${MODEL_TYPE:-huggingface}" = "openai_compat" ]; then
    echo "API mode: CPU-only stage_get_cfg (no --gres)."
    SBATCH_CMD+=(
        --cpus-per-task "${API_CFG_CPUS:-4}"
        --mem "${API_CFG_MEM:-8G}"
        --time "${API_CFG_TIME:-00:20:00}"
    )
else
    echo "Non-API mode: requesting GPU resources for stage_get_cfg."
    SBATCH_CMD+=(
        --cpus-per-task "${HF_CFG_CPUS:-32}"
        --mem "${HF_CFG_MEM:-256G}"
        --gres "${HF_CFG_GRES:-gpu:4}"
        --time "${HF_CFG_TIME:-00:15:00}"
    )
fi

SBATCH_CMD+=(scripts/stages/stage_get_cfg.slurm)
"${SBATCH_CMD[@]}"

echo ""
echo "Pipeline job submitted!"
echo "  Monitor with: squeue -u \$USER"
echo "  Check logs in: $LOG_DIR/"
