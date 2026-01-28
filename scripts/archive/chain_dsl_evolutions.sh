#!/bin/bash
# Script to automatically chain pipeline jobs for DSL evolution
# Usage: ./chain_dsl_evolutions.sh [MAX_DSL_EVOLUTIONS] [EXPERIMENT_DIR]
#
# This script submits jobs one at a time, checking exit codes between them.
# Jobs only continue if the previous job exited with 0 (success) or 100 (DSL evolution).

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default max DSL evolutions
MAX_DSL_EVOLUTIONS="${1:-2}"
EXPERIMENT_DIR_ARG="$2"

# Check if help requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [MAX_DSL_EVOLUTIONS] [EXPERIMENT_DIR]"
    echo ""
    echo "Submits pipeline jobs for DSL evolution in a chain."
    echo "Each job waits for the previous one and only continues if exit code is 0 or 100."
    echo ""
    echo "Arguments:"
    echo "  MAX_DSL_EVOLUTIONS  - Maximum number of DSL evolution rounds (default: 2)"
    echo "  EXPERIMENT_DIR      - Experiment directory (optional, will auto-generate if not set)"
    echo ""
    echo "Examples:"
    echo "  $0 3"
    echo "  $0 3 experiment_unified_20251215_143209_job2663241"
    echo ""
    echo "The script will submit jobs sequentially, checking exit codes between them."
    echo "Jobs will automatically resume from checkpoint if DSL evolution occurred (exit code 100)."
    exit 0
fi

# Validate MAX_DSL_EVOLUTIONS is a number
if ! [[ "$MAX_DSL_EVOLUTIONS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: MAX_DSL_EVOLUTIONS must be a number, got: $MAX_DSL_EVOLUTIONS"
    exit 1
fi

if [ "$MAX_DSL_EVOLUTIONS" -lt 1 ]; then
    echo "ERROR: MAX_DSL_EVOLUTIONS must be at least 1"
    exit 1
fi

# Export EXPERIMENT_DIR if provided
if [ -n "$EXPERIMENT_DIR_ARG" ]; then
    export EXPERIMENT_DIR="$EXPERIMENT_DIR_ARG"
    echo "Using provided EXPERIMENT_DIR: $EXPERIMENT_DIR"
fi

# Export other environment variables if they're set (preserve them)
if [ -n "$SPEC_FILE" ]; then
    export SPEC_FILE
fi
if [ -n "$TASKS" ]; then
    export TASKS
fi
if [ -n "$MAX_FUNCTION_EVOLUTIONS" ]; then
    export MAX_FUNCTION_EVOLUTIONS
fi
# Set MAX_DSL_EVOLUTIONS for the pipeline
export MAX_DSL_EVOLUTIONS
if [ -n "$RECIPES_PATH" ]; then
    export RECIPES_PATH
fi
if [ -n "$HINTS_PATH" ]; then
    export HINTS_PATH
fi

SLURM_SCRIPT="$SCRIPT_DIR/run_integrated_pipeline.slurm"

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "Submitting DSL Evolution Job Chain"
echo "=========================================="
echo "Max DSL Evolutions: $MAX_DSL_EVOLUTIONS"
if [ -n "$EXPERIMENT_DIR" ]; then
    echo "Experiment Directory: $EXPERIMENT_DIR"
fi
echo ""
echo "Jobs will be submitted sequentially."
echo "Each job will wait for the previous one and only continue if exit code is 0 or 100."
echo "=========================================="
echo ""

# Submit first job
echo "Submitting job 1/$MAX_DSL_EVOLUTIONS (initial run)..."
CURRENT_JOB_ID=$(sbatch --parsable --export=ALL "$SLURM_SCRIPT" 2>&1)
SBATCH_EXIT_CODE=$?

if [ $SBATCH_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to submit first job"
    echo "sbatch output: $CURRENT_JOB_ID"
    exit 1
fi

if ! [[ "$CURRENT_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Failed to get valid job ID from sbatch"
    echo "sbatch output: $CURRENT_JOB_ID"
    exit 1
fi

echo "  Job 1 submitted: $CURRENT_JOB_ID"
echo "  Waiting for job to complete..."

# Wait for first job and check exit code
JOB_EXIT_CODE=""
while [ -z "$JOB_EXIT_CODE" ]; do
    sleep 10
    # Check if job is still running
    if ! squeue -j "$CURRENT_JOB_ID" -h &>/dev/null; then
        # Job finished, get exit code
        JOB_EXIT_CODE=$(sacct -j "$CURRENT_JOB_ID" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
        if [ -n "$JOB_EXIT_CODE" ]; then
            break
        fi
    fi
done

echo "  Job 1 completed with exit code: $JOB_EXIT_CODE"

# Submit remaining jobs in chain
for i in $(seq 2 $MAX_DSL_EVOLUTIONS); do
    # Check if we should continue (exit code 0 = success, 100 = DSL evolution)
    if [ "$JOB_EXIT_CODE" != "0" ] && [ "$JOB_EXIT_CODE" != "100" ]; then
        echo ""
        echo "Previous job exited with code $JOB_EXIT_CODE (not 0 or 100)"
        echo "Stopping chain. Remaining jobs will not be submitted."
        break
    fi
    
    # Check if checkpoint exists (for DSL evolution case)
    if [ "$JOB_EXIT_CODE" = "100" ]; then
        if [ -z "$EXPERIMENT_DIR" ]; then
            echo "WARNING: EXPERIMENT_DIR not set, cannot verify checkpoint"
        else
            CHECKPOINT_FILE="$PROJECT_ROOT/$EXPERIMENT_DIR/checkpoint.json"
            if [ ! -f "$CHECKPOINT_FILE" ]; then
                echo "WARNING: Exit code 100 but no checkpoint found at $CHECKPOINT_FILE"
            else
                echo "  Checkpoint found - DSL evolution occurred"
            fi
        fi
    fi
    
    # Submit next job with dependency
    echo ""
    echo "Submitting job $i/$MAX_DSL_EVOLUTIONS (depends on job $CURRENT_JOB_ID)..."
    
    # Use afterany so job continues after previous job completes (regardless of exit code)
    NEW_JOB_ID=$(sbatch --parsable --dependency=afterany:$CURRENT_JOB_ID --export=ALL "$SLURM_SCRIPT" 2>&1)
    SBATCH_EXIT_CODE=$?
    
    if [ $SBATCH_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Failed to submit job $i"
        echo "sbatch output: $NEW_JOB_ID"
        exit 1
    fi
    
    if ! [[ "$NEW_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to get valid job ID from sbatch for job $i"
        echo "sbatch output: $NEW_JOB_ID"
        exit 1
    fi
    
    echo "  Job $i submitted: $NEW_JOB_ID (depends on $CURRENT_JOB_ID)"
    echo "  Waiting for job to complete..."
    
    CURRENT_JOB_ID="$NEW_JOB_ID"
    
    # Wait for this job to complete
    JOB_EXIT_CODE=""
    while [ -z "$JOB_EXIT_CODE" ]; do
        sleep 10
        if ! squeue -j "$CURRENT_JOB_ID" -h &>/dev/null; then
            JOB_EXIT_CODE=$(sacct -j "$CURRENT_JOB_ID" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
            if [ -n "$JOB_EXIT_CODE" ]; then
                break
            fi
        fi
    done
    
    echo "  Job $i completed with exit code: $JOB_EXIT_CODE"
done

echo ""
echo "=========================================="
echo "Job Chain Complete!"
echo "=========================================="
echo ""
echo "Monitor remaining jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs with:"
echo "  tail -f $PROJECT_ROOT/scripts/log/unified_pipeline_*.out"
echo "=========================================="
