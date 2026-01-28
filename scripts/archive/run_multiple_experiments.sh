#!/bin/bash
# Helper script to run multiple experiments in parallel
# Usage: 
#   ./scripts/run_multiple_experiments.sh <experiment1_dir> <experiment2_dir> ...
#   ./scripts/run_multiple_experiments.sh <config_file>
#   ./scripts/run_multiple_experiments.sh --auto-generate <count> [--prefix <prefix>]

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Function to generate a unique experiment directory name
generate_experiment_dir() {
    local prefix="${1:-experiment}"
    local seq="${2:-}"
    
    # Use timestamp + sequence number + random component for uniqueness
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local random=$(openssl rand -hex 4 2>/dev/null || echo $(shuf -i 1000-9999 -n 1))
    
    if [ -n "$seq" ]; then
        echo "${prefix}_${timestamp}_${seq}_${random}"
    else
        echo "${prefix}_${timestamp}_${random}"
    fi
}

# Function to submit a single experiment
submit_experiment() {
    local exp_dir="$1"
    
    # If exp_dir is empty or "AUTO", generate a unique experiment directory name
    if [ -z "$exp_dir" ] || [ "$exp_dir" = "AUTO" ]; then
        exp_dir=$(generate_experiment_dir "experiment" "")
    fi
    
    local exp_name=$(basename "$exp_dir")
    
    echo "=========================================="
    echo "Submitting experiment: $exp_name"
    echo "  Experiment directory: $exp_dir"
    echo "=========================================="
    
    # Always pass EXPERIMENT_DIR so the chaining script can use it for job naming
    # The Python script will create the directory if it doesn't exist
    sbatch \
        --job-name="${exp_name}_get_cfg" \
        --export="ALL,EXPERIMENT_DIR=$exp_dir" \
        scripts/stages/stage_get_cfg.slurm
    
    echo "  ✓ Submitted initial job for $exp_name"
    echo ""
}

# Main logic
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment1_dir> [experiment2_dir] ..."
    echo "   OR: $0 <config_file>"
    echo "   OR: $0 --auto-generate <count> [--prefix <prefix>]"
    echo ""
    echo "Options:"
    echo "  <experiment_dir>     : Use specific experiment directory"
    echo "  <config_file>        : Read experiment directories from file (one per line)"
    echo "  --auto-generate <N> : Auto-generate N experiment directories"
    echo "  --prefix <prefix>   : Prefix for auto-generated directories (default: 'experiment')"
    echo ""
    echo "Each experiment will run independently with its own state file and job names."
    exit 1
fi

# Check for auto-generate mode
if [ "$1" = "--auto-generate" ] || [ "$1" = "-a" ]; then
    count="${2:-1}"
    prefix="experiment"
    
    # Parse optional prefix
    if [ "$#" -ge 4 ] && [ "$3" = "--prefix" ]; then
        prefix="$4"
    fi
    
    echo "Auto-generating $count experiment(s) with prefix '$prefix'..."
    echo ""
    
    for i in $(seq 1 "$count"); do
        exp_dir=$(generate_experiment_dir "$prefix" "$i")
        submit_experiment "$exp_dir"
        # Small delay to ensure unique timestamps
        sleep 1
    done
elif [ -f "$1" ]; then
    # Check if first argument is a file (config file)
    echo "Reading experiment directories from config file: $1"
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        line=$(echo "$line" | sed 's/#.*$//' | xargs)
        if [ -n "$line" ]; then
            submit_experiment "$line"
        fi
    done < "$1"
else
    # Treat all arguments as experiment directories
    for exp_dir in "$@"; do
        if [ ! -d "$exp_dir" ] && [ ! -z "${exp_dir:-}" ] && [ "$exp_dir" != "AUTO" ]; then
            echo "Warning: Directory does not exist: $exp_dir (will be created by pipeline)"
        fi
        submit_experiment "$exp_dir"
    done
fi

echo "=========================================="
echo "All experiments submitted!"
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all jobs for an experiment with: scancel --job-name=<exp_name>_*"
echo "=========================================="

