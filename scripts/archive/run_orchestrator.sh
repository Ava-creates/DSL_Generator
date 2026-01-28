#!/bin/bash
# Wrapper script to run the orchestrator
# Can run in tmux on login node (for short runs) or submit as SLURM job (recommended for long runs)
# Usage: bash scripts/run_orchestrator.sh [OPTIONS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-pipeline_orchestrator}"
USE_TMUX="${USE_TMUX:-true}"
SPEC_FILE="${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}"

# Parse arguments
EXPERIMENT_DIR=""
# Don't reset SPEC_FILE - keep the default if not provided
TASKS=""
MAX_DSL_EVOLUTIONS=2
MAX_FUNCTION_EVOLUTIONS=1
SKIP_CFG_GENERATION=false
CFG_OUTPUT_FILE=""
MAX_CFG_RETRIES=10
RECIPES_PATH="craft/resources/recipes.yaml"
HINTS_PATH="craft/resources/hints.yaml"
MAX_ATTEMPTS=1
MODEL_TYPE="huggingface"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run the pipeline orchestrator. This script manages the pipeline by submitting
separate jobs for each stage. It can run in a tmux session for long-running
pipelines.

Options:
    --experiment_dir DIR          Experiment directory (optional, auto-generated if not provided)
    --spec_file FILE              Specification file (default: prompt_specifications/specification_with_updated_nld.txt)
    --tasks TASK1 [TASK2 ...]     Tasks to solve (required)
    --max_dsl_evolutions N        Max DSL evolution rounds (default: 2)
    --max_function_evolutions N   Max function evolution rounds (default: 3)
    --skip_cfg_generation         Skip CFG generation
    --cfg_output_file FILE        File to load CFG from
    --max_cfg_retries N           Max CFG generation retries (default: 10)
    --recipes_path PATH           Path to recipes YAML (default: craft/resources/recipes.yaml)
    --hints_path PATH             Path to hints YAML (default: craft/resources/hints.yaml)
    --max_attempts N              Max attempts per task (default: 1)
    --model_type TYPE             Model type: huggingface, ollama, gemini (default: huggingface)
    --no-tmux                     Don't use tmux session
    --tmux-session NAME           Tmux session name (default: pipeline_orchestrator)
    -h, --help                    Show this help message

Examples:
    # Run with auto-generated experiment directory (recommended)
    $0 --tasks "make[stick]" "get[gem]"

    # Run with custom experiment directory
    $0 --experiment_dir experiment_001 \\
       --tasks "make[stick]" "get[gem]"

    # Run without tmux
    $0 --no-tmux --tasks "make[stick]" "get[gem]"

    # Run with tasks from JSON file
    $0 --tasks config/task_config.json

    # Run with custom spec file
    $0 --spec_file prompt_specifications/custom_spec.txt \\
       --tasks "make[stick]" "get[gem]"

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir)
            EXPERIMENT_DIR="$2"
            shift 2
            ;;
        --spec_file)
            SPEC_FILE="$2"
            shift 2
            ;;
        --tasks)
            shift
            TASKS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TASKS+=("$1")
                shift
            done
            ;;
        --max_dsl_evolutions)
            MAX_DSL_EVOLUTIONS="$2"
            shift 2
            ;;
        --max_function_evolutions)
            MAX_FUNCTION_EVOLUTIONS="$2"
            shift 2
            ;;
        --skip_cfg_generation)
            SKIP_CFG_GENERATION=true
            shift
            ;;
        --cfg_output_file)
            CFG_OUTPUT_FILE="$2"
            shift 2
            ;;
        --max_cfg_retries)
            MAX_CFG_RETRIES="$2"
            shift 2
            ;;
        --recipes_path)
            RECIPES_PATH="$2"
            shift 2
            ;;
        --hints_path)
            HINTS_PATH="$2"
            shift 2
            ;;
        --max_attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --no-tmux)
            USE_TMUX=false
            shift
            ;;
        --tmux-session)
            TMUX_SESSION_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Error: --tasks is required" >&2
    show_usage
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f "new_dsl_env/bin/activate" ]; then
    source new_dsl_env/bin/activate
    echo "Activated virtual environment: new_dsl_env"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment: venv"
else
    echo "Warning: No virtual environment found. Trying to continue anyway..." >&2
fi

# Build command (use -u flag for unbuffered output)
CMD="python -u scripts/orchestrator.py"
# Only add --experiment_dir if it's not empty
if [ -n "$EXPERIMENT_DIR" ]; then
    CMD="$CMD --experiment_dir \"$EXPERIMENT_DIR\""
fi
CMD="$CMD --spec_file \"$SPEC_FILE\""
CMD="$CMD --tasks ${TASKS[@]}"
CMD="$CMD --max_dsl_evolutions $MAX_DSL_EVOLUTIONS"
CMD="$CMD --max_function_evolutions $MAX_FUNCTION_EVOLUTIONS"
if [ "$SKIP_CFG_GENERATION" = true ]; then
    CMD="$CMD --skip_cfg_generation"
fi
if [ -n "$CFG_OUTPUT_FILE" ]; then
    CMD="$CMD --cfg_output_file \"$CFG_OUTPUT_FILE\""
fi
CMD="$CMD --max_cfg_retries $MAX_CFG_RETRIES"
CMD="$CMD --recipes_path \"$RECIPES_PATH\""
CMD="$CMD --hints_path \"$HINTS_PATH\""
CMD="$CMD --max_attempts $MAX_ATTEMPTS"
CMD="$CMD --model_type $MODEL_TYPE"

# Create log directory
mkdir -p scripts/log

# Run in tmux or directly
if [ "$USE_TMUX" = true ]; then
    # Check if tmux is available
    if ! command -v tmux &> /dev/null; then
        echo "Warning: tmux not found, running without tmux" >&2
        USE_TMUX=false
    fi
fi

if [ "$USE_TMUX" = true ]; then
    # Check if session already exists
    if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
        echo "Tmux session '$TMUX_SESSION_NAME' already exists."
        echo "Attach with: tmux attach -t $TMUX_SESSION_NAME"
        echo "Or kill it first with: tmux kill-session -t $TMUX_SESSION_NAME"
        exit 1
    fi
    
    # Create new tmux session and run orchestrator
    LOG_FILE="scripts/log/orchestrator_$(date +%Y%m%d_%H%M%S).log"
    echo "Starting orchestrator in tmux session '$TMUX_SESSION_NAME'"
    echo "Log file: $LOG_FILE"
    echo "Attach with: tmux attach -t $TMUX_SESSION_NAME"
    echo ""
    
    # Set PYTHONUNBUFFERED to ensure output is not buffered
    export PYTHONUNBUFFERED=1
    
    # Prepare activation command for tmux session
    if [ -f "new_dsl_env/bin/activate" ]; then
        ACTIVATE_CMD="source new_dsl_env/bin/activate && "
    elif [ -f "venv/bin/activate" ]; then
        ACTIVATE_CMD="source venv/bin/activate && "
    else
        ACTIVATE_CMD=""
    fi
    
    tmux new-session -d -s "$TMUX_SESSION_NAME" -c "$PROJECT_ROOT" \
        "bash -c 'export PYTHONUNBUFFERED=1; ${ACTIVATE_CMD}$CMD 2>&1 | tee $LOG_FILE'"
    
    echo "Orchestrator started in tmux session '$TMUX_SESSION_NAME'"
    echo "Monitor with: tail -f $LOG_FILE"
    echo "Or attach to tmux: tmux attach -t $TMUX_SESSION_NAME"
else
    # Run directly
    echo "Running orchestrator..."
    export PYTHONUNBUFFERED=1
    eval $CMD
fi

