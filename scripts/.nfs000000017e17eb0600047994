#!/bin/bash
# Unified pipeline submission script
# Usage: ./scripts/run_pipeline.sh [--background]
# 
# This script automates the full pipeline workflow:
# 1. Prompts for configuration parameters
# 2. Submits first iteration job
# 3. Monitors job completion
# 4. Automatically chains CFG version jobs when DSL evolves (exit code 100)
# 5. Continues until pipeline completes (exit code 0) or fails
#
# Note: The script will monitor jobs until completion. For long-running jobs:
#   - Use screen/tmux: screen -S pipeline or tmux new -s pipeline
#   - Or use --background flag to run in background (outputs to log file)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check for background flag
BACKGROUND_MODE=false
if [[ "${1:-}" == "--background" ]] || [[ "${1:-}" == "-b" ]]; then
    BACKGROUND_MODE=true
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to prompt for input with default value
prompt_with_default() {
    local prompt_text="$1"
    local default_value="$2"
    local var_name="$3"
    
    if [ -n "$default_value" ]; then
        read -p "$prompt_text [$default_value]: " input
        eval "$var_name=\"\${input:-$default_value}\""
    else
        read -p "$prompt_text: " input
        eval "$var_name=\"\$input\""
    fi
}

# Function to prompt for optional input
prompt_optional() {
    local prompt_text="$1"
    local var_name="$2"
    
    read -p "$prompt_text (press Enter to skip): " input
    if [ -n "$input" ]; then
        eval "$var_name=\"\$input\""
    else
        eval "$var_name=\"\""
    fi
}

# Function to collect all inputs interactively
prompt_for_inputs() {
    echo "=========================================="
    echo "Pipeline Configuration"
    echo "=========================================="
    echo ""
    
    # Required: Experiment directory (can be empty to auto-generate)
    prompt_optional "Experiment directory (leave empty to auto-generate)" EXPERIMENT_DIR
    
    # Optional parameters with defaults
    prompt_with_default "Spec file" "prompt_specifications/specification_for_cfg.txt" SPEC_FILE
    prompt_with_default "Tasks config" "config/task_config.json" TASKS
    prompt_with_default "Max function evolutions" "3" MAX_FUNCTION_EVOLUTIONS
    prompt_with_default "Max DSL evolutions" "2" MAX_DSL_EVOLUTIONS
    prompt_with_default "Recipes path" "craft/resources/recipes.yaml" RECIPES_PATH
    prompt_with_default "Hints path" "craft/resources/hints.yaml" HINTS_PATH
    prompt_with_default "Model type" "huggingface" MODEL_TYPE
    
    echo ""
    echo "=========================================="
    echo "Configuration Summary"
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
    
    read -p "Proceed with these settings? (y/n) " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    
    echo ""
    echo "Proceeding with job submission..."
    
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
}

# Function to submit first iteration job
submit_first_job() {
    echo "" >&2
    echo "==========================================" >&2
    echo "Submitting First Iteration Job" >&2
    echo "==========================================" >&2
    echo "This job will run for 2 days or until first DSL iteration completes" >&2
    echo "" >&2
    
    local job_id
    job_id=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/run_first_iteration.slurm" 2>&1)
    local sbatch_exit=$?
    
    if [ $sbatch_exit -ne 0 ]; then
        echo -e "${RED}✗ Failed to submit job${NC}" >&2
        echo "sbatch output: $job_id" >&2
        return 1
    fi
    
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo -e "${GREEN}✓ First iteration job submitted: $job_id${NC}" >&2
        echo "$job_id"
        return 0
    else
        echo -e "${RED}✗ Failed to get valid job ID${NC}" >&2
        echo "sbatch output: $job_id" >&2
        return 1
    fi
}

# Function to wait for job completion and get exit code
wait_for_job() {
    local job_id="$1"
    local job_name="$2"
    
    echo ""
    echo "Monitoring $job_name (job ID: $job_id)..."
    echo "  View logs: tail -f $SCRIPT_DIR/log/${job_name}_${job_id}.out"
    echo ""
    
    # Wait for job to finish
    while true; do
        sleep 10
        
        # Check if job is still in queue
        if ! squeue -j "$job_id" -h &>/dev/null; then
            # Job finished, get exit code
            local exit_code
            exit_code=$(sacct -j "$job_id" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
            
            if [ -n "$exit_code" ]; then
                echo ""
                echo "$job_name completed with exit code: $exit_code"
                echo "$exit_code"
                return 0
            fi
            
            # If sacct doesn't have the exit code yet, wait a bit more
            sleep 5
            exit_code=$(sacct -j "$job_id" --format=ExitCode -n -P 2>/dev/null | head -1 | cut -d: -f1)
            if [ -n "$exit_code" ]; then
                echo ""
                echo "$job_name completed with exit code: $exit_code"
                echo "$exit_code"
                return 0
            fi
        fi
    done
}

# Function to get EXPERIMENT_DIR from first job's output or checkpoint
get_experiment_dir() {
    local job_id="$1"
    
    # If EXPERIMENT_DIR was set, use it
    if [ -n "${EXPERIMENT_DIR:-}" ]; then
        echo "$EXPERIMENT_DIR"
        return 0
    fi
    
    # Try to extract from log file
    local log_file="$SCRIPT_DIR/log/first_iteration_${job_id}.out"
    if [ -f "$log_file" ]; then
        local extracted_dir
        extracted_dir=$(grep -oP 'Generated experiment directory: \K[^\s]+' "$log_file" 2>/dev/null || echo "")
        if [ -n "$extracted_dir" ]; then
            echo "$extracted_dir"
            return 0
        fi
    fi
    
    # Try to find most recent experiment directory
    local recent_dir
    recent_dir=$(find "$PROJECT_ROOT" -maxdepth 1 -type d -name "experiment_unified_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$recent_dir" ]; then
        echo "$(basename "$recent_dir")"
        return 0
    fi
    
    return 1
}

# Function to submit CFG version job
submit_cfg_job() {
    local experiment_dir="$1"
    
    echo "" >&2
    echo "==========================================" >&2
    echo "Submitting CFG Version Job" >&2
    echo "==========================================" >&2
    echo "Experiment Directory: $experiment_dir" >&2
    echo "" >&2
    
    # Verify checkpoint exists
    local checkpoint_file="$PROJECT_ROOT/$experiment_dir/checkpoint.json"
    if [ ! -f "$checkpoint_file" ]; then
        echo -e "${RED}✗ Checkpoint file not found: $checkpoint_file${NC}" >&2
        return 1
    fi
    
    # Read CFG version from checkpoint
    local cfg_version
    cfg_version=$(python3 -c "import json; print(json.load(open('$checkpoint_file')).get('cfg_version', 0))" 2>/dev/null || echo "0")
    echo "Processing CFG version: $cfg_version" >&2
    echo "" >&2
    
    # Submit job with EXPERIMENT_DIR set
    local job_id
    job_id=$(sbatch --parsable --export=ALL,EXPERIMENT_DIR="$experiment_dir" "$SCRIPT_DIR/run_cfg_version.slurm" 2>&1)
    local sbatch_exit=$?
    
    if [ $sbatch_exit -ne 0 ]; then
        echo -e "${RED}✗ Failed to submit CFG version job${NC}" >&2
        echo "sbatch output: $job_id" >&2
        return 1
    fi
    
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo -e "${GREEN}✓ CFG version job submitted: $job_id${NC}" >&2
        echo "$job_id"
        return 0
    else
        echo -e "${RED}✗ Failed to get valid job ID${NC}" >&2
        echo "sbatch output: $job_id" >&2
        return 1
    fi
}

# Main function
main() {
    # Warn if not in screen/tmux and not in background mode
    if [ "$BACKGROUND_MODE" = false ]; then
        if [ -z "${STY:-}" ] && [ -z "${TMUX:-}" ]; then
            echo ""
            echo -e "${YELLOW}⚠ Warning: Not running in screen/tmux session${NC}"
            echo "This script will monitor jobs until completion (may take days)."
            echo "If you disconnect, the script will stop."
            echo ""
            echo "Recommended: Run in screen or tmux:"
            echo "  screen -S pipeline ./scripts/run_pipeline.sh"
            echo "  # Then detach with: Ctrl+A, D"
            echo ""
            echo "Or run in background:"
            echo "  nohup ./scripts/run_pipeline.sh > pipeline.log 2>&1 &"
            echo ""
            read -p "Continue anyway? (y/n) " continue_anyway
            if [[ ! "$continue_anyway" =~ ^[Yy]$ ]]; then
                echo "Aborted. Please run in screen/tmux or with nohup."
                exit 0
            fi
            echo ""
        fi
    fi
    
    # If background mode, redirect output to log file
    if [ "$BACKGROUND_MODE" = true ]; then
        LOG_FILE="$PROJECT_ROOT/pipeline_monitor_$(date +%Y%m%d_%H%M%S).log"
        exec > >(tee "$LOG_FILE")
        exec 2>&1
        echo "Running in background mode. Log file: $LOG_FILE"
        echo ""
    fi
    
    # Collect inputs
    prompt_for_inputs
    
    # Submit first iteration job
    local first_job_id
    if ! first_job_id=$(submit_first_job); then
        exit 1
    fi
    
    if [ -z "$first_job_id" ] || [[ ! "$first_job_id" =~ ^[0-9]+$ ]]; then
        echo -e "${RED}✗ Failed to get valid job ID${NC}"
        exit 1
    fi
    
    # Wait for first job to complete
    local exit_code
    exit_code=$(wait_for_job "$first_job_id" "First iteration")
    
    # Get experiment directory (may have been auto-generated)
    local experiment_dir
    experiment_dir=$(get_experiment_dir "$first_job_id")
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Could not determine experiment directory${NC}"
        echo "Please check the job logs and set EXPERIMENT_DIR manually if needed"
        exit 1
    fi
    
    # Export EXPERIMENT_DIR for subsequent jobs
    export EXPERIMENT_DIR="$experiment_dir"
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
        local cfg_job_id
        local cfg_exit_code
        local iteration=1
        
        while true; do
            echo ""
            echo "=========================================="
            echo "CFG Version Iteration $iteration"
            echo "=========================================="
            
            # Submit CFG version job
            if ! cfg_job_id=$(submit_cfg_job "$experiment_dir"); then
                echo -e "${RED}✗ Failed to submit CFG version job${NC}"
                exit 1
            fi
            
            if [ -z "$cfg_job_id" ] || [[ ! "$cfg_job_id" =~ ^[0-9]+$ ]]; then
                echo -e "${RED}✗ Failed to get valid job ID for CFG version job${NC}"
                exit 1
            fi
            
            # Wait for job to complete
            cfg_exit_code=$(wait_for_job "$cfg_job_id" "CFG version")
            
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
                echo "  export EXPERIMENT_DIR=$experiment_dir"
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
        echo "Experiment directory: $experiment_dir"
        echo ""
        echo "To resume manually:"
        if [ -n "$experiment_dir" ]; then
            echo "  export EXPERIMENT_DIR=$experiment_dir"
            echo "  sbatch $SCRIPT_DIR/run_cfg_version.slurm"
        else
            echo "  ./scripts/submit_first_job.sh"
        fi
        exit 2
    fi
}

# Run main function
main "$@"

