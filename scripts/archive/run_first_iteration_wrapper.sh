#!/bin/bash
# Wrapper script for first iteration job
# This job runs for 2 days or until first DSL iteration completes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}"
TASKS="${TASKS:-config/task_config.json}"
MAX_FUNCTION_EVOLUTIONS="${MAX_FUNCTION_EVOLUTIONS:-1}"
MAX_DSL_EVOLUTIONS="${MAX_DSL_EVOLUTIONS:-2}"
RECIPES_PATH="${RECIPES_PATH:-craft/resources/recipes.yaml}"
HINTS_PATH="${HINTS_PATH:-craft/resources/hints.yaml}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
MODEL_TYPE="${MODEL_TYPE:-huggingface}"

# Generate experiment directory name if not provided
if [ -z "$EXPERIMENT_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    JOB_ID="${SLURM_JOB_ID:-unknown}"
    EXPERIMENT_DIR="experiment_unified_${TIMESTAMP}_job${JOB_ID}"
    export EXPERIMENT_DIR
    echo "Generated experiment directory: $EXPERIMENT_DIR"
fi

echo "=========================================="
echo "First Iteration Job"
echo "=========================================="
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Spec File: $SPEC_FILE"
echo "Tasks: $TASKS"
echo "Max Function Evolutions: $MAX_FUNCTION_EVOLUTIONS"
echo "Max DSL Evolutions: $MAX_DSL_EVOLUTIONS"
echo "Max Attempts: $MAX_ATTEMPTS"
echo "Time Limit: 2 days"
echo "=========================================="
echo ""
set +e
# Run unified pipeline
# This will run until first DSL iteration completes or 2 days elapses
python3 -m src.pipeline.unified_pipeline \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "$SPEC_FILE" \
    --tasks "$TASKS" \
    --max_dsl_evolutions "$MAX_DSL_EVOLUTIONS" \
    --max_function_evolutions "$MAX_FUNCTION_EVOLUTIONS" \
    --recipes_path "$RECIPES_PATH" \
    --hints_path "$HINTS_PATH" \
    --max_attempts "$MAX_ATTEMPTS" \
    --model_type "$MODEL_TYPE"

EXIT_CODE=$?
set -e
echo ""
echo "=========================================="
echo "First Iteration Job Completed"
echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo ""

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tasks solved! Pipeline complete."
    exit 0
elif [ $EXIT_CODE -eq 100 ]; then
    echo "DSL evolved. Checkpoint saved."
    
    # Read CFG version from checkpoint
    CHECKPOINT_FILE="$PROJECT_ROOT/$EXPERIMENT_DIR/checkpoint.json"
    CFG_VERSION=$(python3 -c "import json; print(json.load(open('$CHECKPOINT_FILE')).get('cfg_version', 0))" 2>/dev/null || echo "0")
    
    echo "CFG version: $CFG_VERSION"
    echo ""
    echo "Submitting next job for CFG version $CFG_VERSION..."
    
    # Submit next job automatically
    NEXT_JOB_ID=$(sbatch --parsable --export=ALL,EXPERIMENT_DIR="$EXPERIMENT_DIR" "$SCRIPT_DIR/run_cfg_version.slurm" 2>&1)
    
    if [[ "$NEXT_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "✓ Next job submitted: $NEXT_JOB_ID"
        echo "  This job will process CFG version $CFG_VERSION"
    else
        echo "⚠ Failed to auto-submit next job. Please submit manually:"
        echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
        echo "  sbatch scripts/run_cfg_version.slurm"
    fi
    
    exit 100
else
    echo "✗ Pipeline failed or time limit reached"
    echo "Exit code: $EXIT_CODE"
    exit $EXIT_CODE
fi

