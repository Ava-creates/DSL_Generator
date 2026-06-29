#!/bin/bash
#SBATCH --job-name=stage_implement_cfg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_implement_cfg_%j.out
#SBATCH --error=scripts/log/stage_implement_cfg_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

source scripts/stages/log_chain_context.sh
log_chain_context "stage_implement_cfg"

# Run the stage script
if [ -n "${FUNC_EVOLUTION_ROUND:-}" ]; then
    FUNC_ROUND_ARG="--func_evolution_round $FUNC_EVOLUTION_ROUND"
else
    FUNC_ROUND_ARG=""
fi

python src/pipeline/stages/stage_implement_cfg.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}" \
    --model_type "${MODEL_TYPE:-huggingface}" \
    --dsl_round "${DSL_ROUND:-0}" \
    $FUNC_ROUND_ARG

# Chain to next stage based on state
if [ $? -eq 0 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

