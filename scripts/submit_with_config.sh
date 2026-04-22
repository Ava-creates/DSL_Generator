#!/bin/bash
# Helper script to submit pipeline with config file
# Usage: ./scripts/submit_with_config.sh [config_file.yaml]

set -e

# Get config file path
CONFIG_FILE="${1:-config/experiment_config.yaml}"

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
echo "  TOTAL_SAMPLES: ${TOTAL_SAMPLES:-1000}"
echo "  MAX_DSL_EVOLUTIONS: ${MAX_DSL_EVOLUTIONS:-2}"
echo "  MAX_FUNCTION_EVOLUTIONS: ${MAX_FUNCTION_EVOLUTIONS:-1}"
echo ""

# Submit the first job
echo "Submitting pipeline job..."
# Use experiment-specific log folder when EXPERIMENT_DIR is set
LOG_DIR="scripts/log"
if [ -n "${EXPERIMENT_DIR:-}" ]; then
    LOG_DIR="scripts/log/$(basename "$EXPERIMENT_DIR")"
fi
mkdir -p "$LOG_DIR"
SBATCH_CMD=(
    sbatch
    --export=ALL,EXPERIMENT_CONFIG="$CONFIG_FILE"
    --output="${LOG_DIR}/stage_get_cfg_%j.out"
    --error="${LOG_DIR}/stage_get_cfg_%j.err"
)

if [ "${MODEL_TYPE:-huggingface}" = "openai_compat" ]; then
    echo "API mode detected for staged pipeline: overriding stage_get_cfg resources to CPU-only."
    SBATCH_CMD+=(
        --cpus-per-task "${API_CFG_CPUS:-4}"
        --mem "${API_CFG_MEM:-8G}"
        --gres "gpu:0"
        --time "${API_CFG_TIME:-00:20:00}"
    )
fi

SBATCH_CMD+=(scripts/stages/stage_get_cfg.slurm)
"${SBATCH_CMD[@]}"

echo ""
echo "Pipeline job submitted!"
echo "  Monitor with: squeue -u \$USER"
echo "  Check logs in: $LOG_DIR/"

