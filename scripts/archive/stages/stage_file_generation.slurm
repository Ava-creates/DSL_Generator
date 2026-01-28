#!/bin/bash
#SBATCH --job-name=stage_file_gen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_file_generation_%j.out
#SBATCH --error=scripts/log/stage_file_generation_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run the stage script
if [ -n "${FUNC_EVOLUTION_ROUND:-}" ]; then
    FUNC_ROUND_ARG="--func_evolution_round $FUNC_EVOLUTION_ROUND"
else
    FUNC_ROUND_ARG=""
fi

python src/pipeline/stages/stage_file_generation.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}" \
    --dsl_round "${DSL_ROUND:-0}" \
    $FUNC_ROUND_ARG

# Chain to next stage based on state
if [ $? -eq 0 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

