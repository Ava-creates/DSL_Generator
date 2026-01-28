#!/bin/bash
# Script to submit the first iteration job
# Usage: bash scripts/submit_first_job.sh [EXPERIMENT_DIR]
#       OR: ./scripts/submit_first_job.sh [EXPERIMENT_DIR]
# 
# NOTE: Do NOT use 'sbatch' on this script - it's a bash script, not a SLURM script!
# This script submits the SLURM job for you.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional: Set experiment directory
if [ -n "${1:-}" ]; then
    export EXPERIMENT_DIR="$1"
    echo "Using experiment directory: $EXPERIMENT_DIR"
fi

# Export other environment variables if needed (optional)
# export SPEC_FILE="prompt_specifications/specification_for_cfg.txt"
# export TASKS="config/task_config.json"
# export MAX_FUNCTION_EVOLUTIONS=3
# export MAX_DSL_EVOLUTIONS=3

echo "=========================================="
echo "Submitting First Iteration Job"
echo "=========================================="
echo "This job will run for 2 days or until first DSL iteration completes"
echo ""

# Submit first job
JOB_ID=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/run_first_iteration.slurm" 2>&1)

if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "✓ First iteration job submitted: $JOB_ID"
    echo ""
    echo "Monitor job with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View logs with:"
    echo "  tail -f scripts/log/first_iteration_${JOB_ID}.out"
    echo ""
    echo "After this job completes:"
    echo "  - If exit code is 100 (DSL evolved), submit next job:"
    echo "    export EXPERIMENT_DIR=<experiment_dir>"
    echo "    sbatch scripts/run_cfg_version.slurm"
    echo "  - If exit code is 0 (all tasks solved), pipeline is complete!"
else
    echo "✗ Failed to submit job"
    echo "sbatch output: $JOB_ID"
    exit 1
fi

