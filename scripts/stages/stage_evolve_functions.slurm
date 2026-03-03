#!/bin/bash
#SBATCH --job-name=stage_evolve_func
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_evolve_functions_%j.out
#SBATCH --error=scripts/log/stage_evolve_functions_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

source scripts/stages/log_chain_context.sh
log_chain_context "stage_evolve_functions"

# Run the stage script
# Build command - failing_tasks is optional now (evolve uses domain template)
CMD="python src/pipeline/stages/stage_evolve_functions.py \
    --experiment_dir \"$EXPERIMENT_DIR\" \
    --spec_file \"$SPEC_FILE\" \
    --model_type \"${MODEL_TYPE:-huggingface}\" \
    --dsl_round \"${DSL_ROUND:-0}\" \
    --func_evolution_round \"${FUNC_EVOLUTION_ROUND:-0}\""

if [ -n "$FAILING_TASKS" ]; then
    CMD="$CMD --failing_tasks $FAILING_TASKS"
fi

eval $CMD

