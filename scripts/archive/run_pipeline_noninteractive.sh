#!/bin/bash
# Non-interactive version of run_pipeline.sh for SLURM jobs
# Usage: Set environment variables and run this script
# Example:
#   export EXPERIMENT_DIR="my_experiment"
#   export SPEC_FILE="prompt_specifications/specification_with_updated_nld.txt"
#   ./scripts/run_pipeline_noninteractive.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set defaults if not provided
EXPERIMENT_DIR="${EXPERIMENT_DIR:-}"
SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}"
TASKS="${TASKS:-config/task_config.json}"
MAX_FUNCTION_EVOLUTIONS="${MAX_FUNCTION_EVOLUTIONS:-3}"
MAX_DSL_EVOLUTIONS="${MAX_DSL_EVOLUTIONS:-2}"
RECIPES_PATH="${RECIPES_PATH:-craft/resources/recipes.yaml}"
HINTS_PATH="${HINTS_PATH:-craft/resources/hints.yaml}"
MODEL_TYPE="${MODEL_TYPE:-huggingface}"

# Export all variables
if [ -n "${EXPERIMENT_DIR:-}" ]; then
    export EXPERIMENT_DIR
fi
export SPEC_FILE
export TASKS
export MAX_FUNCTION_EVOLUTIONS
export MAX_DSL_EVOLUTIONS
export RECIPES_PATH
export HINTS_PATH
export MODEL_TYPE

echo "=========================================="
echo "Pipeline Configuration (Non-Interactive)"
echo "=========================================="
echo "Experiment Directory: ${EXPERIMENT_DIR:-<auto-generate>}"
echo "Spec File: $SPEC_FILE"
echo "Tasks: $TASKS"
echo "Max Function Evolutions: $MAX_FUNCTION_EVOLUTIONS"
echo "Max DSL Evolutions: $MAX_DSL_EVOLUTIONS"
echo "Recipes Path: $RECIPES_PATH"
echo "Hints Path: $HINTS_PATH"
echo "Model Type: $MODEL_TYPE"
echo "=========================================="
echo ""

# Submit first iteration job
echo ""
echo "=========================================="
echo "Submitting First Iteration Job"
echo "=========================================="
echo "This job will run for 2 days or until first DSL iteration completes"
echo ""

first_job_id=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/run_first_iteration.slurm" 2>&1)
sbatch_exit=$?

if [ $sbatch_exit -ne 0 ]; then
    echo -e "${RED}✗ Failed to submit job${NC}" >&2
    echo "sbatch output: $first_job_id" >&2
    exit 1
fi

if [[ ! "$first_job_id" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}✗ Failed to get valid job ID${NC}" >&2
    echo "sbatch output: $first_job_id" >&2
    exit 1
fi

echo -e "${GREEN}✓ First iteration job submitted: $first_job_id${NC}"
echo ""

# Wait for first job to complete
echo "Monitoring First iteration (job ID: $first_job_id)..."
echo "  View logs: tail -f $SCRIPT_DIR/log/first_iteration_${first_job_id}.out"
echo ""

# Wait for job to finish
while true; do
    sleep 10
    
    # Check if job is still in queue
    if ! squeue -j "$first_job_id" -h &>/dev/null; then
        # Job finished, get exit code
        exit_code=$(sacct -j "$first_job_id" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
        
        if [ -n "$exit_code" ]; then
            echo ""
            echo "First iteration completed with exit code: $exit_code"
            break
        fi
        
        # If sacct doesn't have the exit code yet, wait a bit more
        sleep 5
        exit_code=$(sacct -j "$first_job_id" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
        if [ -n "$exit_code" ]; then
            echo ""
            echo "First iteration completed with exit code: $exit_code"
            break
        fi
    fi
done

# Get experiment directory (may have been auto-generated)
if [ -z "${EXPERIMENT_DIR:-}" ]; then
    # Try to extract from log file
    log_file="$SCRIPT_DIR/log/first_iteration_${first_job_id}.out"
    if [ -f "$log_file" ]; then
        EXPERIMENT_DIR=$(grep -oP 'Generated experiment directory: \K[^\s]+' "$log_file" 2>/dev/null || echo "")
    fi
    
    # Try to find most recent experiment directory
    if [ -z "$EXPERIMENT_DIR" ]; then
        recent_dir=$(find "$PROJECT_ROOT" -maxdepth 1 -type d -name "experiment_unified_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$recent_dir" ]; then
            EXPERIMENT_DIR=$(basename "$recent_dir")
        fi
    fi
fi

if [ -z "$EXPERIMENT_DIR" ]; then
    echo -e "${RED}✗ Could not determine experiment directory${NC}" >&2
    echo "Please check the job logs and set EXPERIMENT_DIR manually if needed" >&2
    exit 1
fi

export EXPERIMENT_DIR
echo "Using experiment directory: $EXPERIMENT_DIR"

# Process exit code
if [ "$exit_code" = "0" ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✓ Pipeline Complete!${NC}"
    echo "=========================================="
    echo "All tasks solved successfully."
    echo "Experiment directory: $EXPERIMENT_DIR"
    exit 0
elif [ "$exit_code" = "100" ]; then
    echo ""
    echo -e "${YELLOW}DSL evolved. Checkpoint saved.${NC}"
    echo "Continuing with CFG version jobs..."
    
    # Chain CFG version jobs until completion or failure
    cfg_job_id=""
    cfg_exit_code=""
    iteration=1
    
    while true; do
        echo ""
        echo "=========================================="
        echo "CFG Version Iteration $iteration"
        echo "=========================================="
        
        # Verify checkpoint exists
        checkpoint_file="$PROJECT_ROOT/$EXPERIMENT_DIR/checkpoint.json"
        if [ ! -f "$checkpoint_file" ]; then
            echo -e "${RED}✗ Checkpoint file not found: $checkpoint_file${NC}" >&2
            exit 1
        fi
        
        # Read CFG version from checkpoint
        cfg_version=$(python3 -c "import json; print(json.load(open('$checkpoint_file')).get('cfg_version', 0))" 2>/dev/null || echo "0")
        echo "Processing CFG version: $cfg_version"
        echo ""
        
        # Submit CFG version job
        cfg_job_id=$(sbatch --parsable --export=ALL,EXPERIMENT_DIR="$EXPERIMENT_DIR" "$SCRIPT_DIR/run_cfg_version.slurm" 2>&1)
        sbatch_exit=$?
        
        if [ $sbatch_exit -ne 0 ]; then
            echo -e "${RED}✗ Failed to submit CFG version job${NC}" >&2
            echo "sbatch output: $cfg_job_id" >&2
            exit 1
        fi
        
        if [[ ! "$cfg_job_id" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}✗ Failed to get valid job ID for CFG version job${NC}" >&2
            echo "sbatch output: $cfg_job_id" >&2
            exit 1
        fi
        
        echo -e "${GREEN}✓ CFG version job submitted: $cfg_job_id${NC}"
        echo ""
        echo "Monitoring CFG version (job ID: $cfg_job_id)..."
        echo "  View logs: tail -f $SCRIPT_DIR/log/cfg_version_${cfg_job_id}.out"
        echo ""
        
        # Wait for job to complete
        while true; do
            sleep 10
            if ! squeue -j "$cfg_job_id" -h &>/dev/null; then
                cfg_exit_code=$(sacct -j "$cfg_job_id" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
                if [ -n "$cfg_exit_code" ]; then
                    echo ""
                    echo "CFG version completed with exit code: $cfg_exit_code"
                    break
                fi
                sleep 5
                cfg_exit_code=$(sacct -j "$cfg_job_id" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
                if [ -n "$cfg_exit_code" ]; then
                    echo ""
                    echo "CFG version completed with exit code: $cfg_exit_code"
                    break
                fi
            fi
        done
        
        # Check exit code
        if [ "$cfg_exit_code" = "0" ]; then
            echo ""
            echo "=========================================="
            echo -e "${GREEN}✓ Pipeline Complete!${NC}"
            echo "=========================================="
            echo "All tasks solved successfully."
            echo "Experiment directory: $EXPERIMENT_DIR"
            echo "Total CFG iterations: $iteration"
            exit 0
        elif [ "$cfg_exit_code" = "100" ]; then
            echo ""
            echo -e "${YELLOW}DSL evolved again. Continuing...${NC}"
            iteration=$((iteration + 1))
            # Continue loop to submit next job
        else
            echo ""
            echo "=========================================="
            echo -e "${RED}✗ Pipeline Failed${NC}"
            echo "=========================================="
            echo "CFG version job exited with code: $cfg_exit_code"
            echo "Experiment directory: $EXPERIMENT_DIR"
            echo "To resume manually:"
            echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
            echo "  sbatch $SCRIPT_DIR/run_cfg_version.slurm"
            exit 2
        fi
    done
else
    echo ""
    echo "=========================================="
    echo -e "${RED}✗ Pipeline Failed${NC}"
    echo "=========================================="
    echo "First iteration job exited with code: $exit_code"
    echo "Experiment directory: $EXPERIMENT_DIR"
    echo ""
    echo "To resume manually:"
    if [ -n "$EXPERIMENT_DIR" ]; then
        echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
        echo "  sbatch $SCRIPT_DIR/run_cfg_version.slurm"
    else
        echo "  ./scripts/submit_first_job.sh"
    fi
    exit 2
fi

