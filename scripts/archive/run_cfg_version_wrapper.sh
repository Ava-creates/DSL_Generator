#!/bin/bash
# Wrapper script for processing a specific CFG version
# This job processes one CFG version and resubmits if DSL evolves

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Required: EXPERIMENT_DIR must be set
if [ -z "${EXPERIMENT_DIR:-}" ]; then
    echo "ERROR: EXPERIMENT_DIR environment variable must be set"
    echo "Usage: export EXPERIMENT_DIR=<experiment_dir> && sbatch scripts/run_cfg_version.slurm"
    exit 1
fi

EXPERIMENT_DIR="$EXPERIMENT_DIR"
CHECKPOINT_FILE="$PROJECT_ROOT/$EXPERIMENT_DIR/checkpoint.json"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT_FILE"
    echo "Please run the first iteration job first"
    exit 1
fi

# Read CFG version from checkpoint
CFG_VERSION=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('cfg_version', 0))" 2>/dev/null || echo "0")

# Default values (will be overridden by checkpoint if resuming)
SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_for_cfg.txt}"
TASKS="${TASKS:-config/task_config.json}"
MAX_FUNCTION_EVOLUTIONS="${MAX_FUNCTION_EVOLUTIONS:-3}"
MAX_DSL_EVOLUTIONS="${MAX_DSL_EVOLUTIONS:-2}"
RECIPES_PATH="${RECIPES_PATH:-craft/resources/recipes.yaml}"
HINTS_PATH="${HINTS_PATH:-craft/resources/hints.yaml}"
MODEL_TYPE="${MODEL_TYPE:-huggingface}"

echo "=========================================="
echo "Processing CFG Version Job"
echo "=========================================="
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "CFG Version: $CFG_VERSION"
echo "Checkpoint File: $CHECKPOINT_FILE"
echo "=========================================="
echo ""

# Run unified pipeline with resume from checkpoint
python3 -m src.pipeline.unified_pipeline \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "$SPEC_FILE" \
    --tasks "$TASKS" \
    --max_dsl_evolutions "$MAX_DSL_EVOLUTIONS" \
    --max_function_evolutions "$MAX_FUNCTION_EVOLUTIONS" \
    --recipes_path "$RECIPES_PATH" \
    --hints_path "$HINTS_PATH" \
    --model_type "$MODEL_TYPE" \
    --resume_from_checkpoint

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "CFG Version Job Completed"
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "CFG Version: $CFG_VERSION"
echo ""

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tasks solved! Pipeline complete."
    exit 0
elif [ $EXIT_CODE -eq 100 ]; then
    echo "DSL evolved. Checkpoint saved."
    
    # Read new CFG version from updated checkpoint
    NEW_CFG_VERSION=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('cfg_version', 0))" 2>/dev/null || echo "0")
    
    echo "New CFG version: $NEW_CFG_VERSION"
    echo ""
    echo "Submitting next job for CFG version $NEW_CFG_VERSION..."
    
    # Submit next job automatically
    NEXT_JOB_ID=$(sbatch --parsable --export=ALL,EXPERIMENT_DIR="$EXPERIMENT_DIR" "$SCRIPT_DIR/run_cfg_version.slurm" 2>&1)
    
    if [[ "$NEXT_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "✓ Next job submitted: $NEXT_JOB_ID"
        echo "  This job will process CFG version $NEW_CFG_VERSION"
    else
        echo "⚠ Failed to auto-submit next job. Please submit manually:"
        echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
        echo "  sbatch scripts/run_cfg_version.slurm"
    fi
    
    exit 100
else
    echo "✗ Pipeline failed or time limit reached"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "To resume, run:"
    echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
    echo "  sbatch scripts/run_cfg_version.slurm"
    exit $EXIT_CODE
fi

