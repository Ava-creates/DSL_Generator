#!/bin/bash
#SBATCH --job-name=stage_evolve_dsl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_evolve_dsl_%j.out
#SBATCH --error=scripts/log/stage_evolve_dsl_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run the stage script
python src/pipeline/stages/stage_evolve_dsl.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --failing_tasks $FAILING_TASKS \
    --recipes_path "${RECIPES_PATH:-craft/resources/recipes.yaml}" \
    --max_retries "${MAX_DSL_RETRIES:-10}"

# Chain to next stage based on state
if [ $? -eq 0 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

