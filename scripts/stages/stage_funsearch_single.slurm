#!/bin/bash
#SBATCH --job-name=funsearch_%x
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_funsearch_%x_%j.out
#SBATCH --error=scripts/log/stage_funsearch_%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

source scripts/stages/log_chain_context.sh
log_chain_context "stage_funsearch_single"

# Set PyTorch CUDA allocation config to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the stage script
if [ -n "${FUNC_EVOLUTION_ROUND:-}" ]; then
    FUNC_ROUND_ARG="--func_evolution_round $FUNC_EVOLUTION_ROUND"
else
    FUNC_ROUND_ARG=""
fi

python src/pipeline/stages/stage_funsearch_single.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "$SPEC_FILE" \
    --function_name "$FUNCTION_NAME" \
    --model_type "${MODEL_TYPE:-huggingface}" \
    --dsl_round "${DSL_ROUND:-0}" \
    --total_samples "${TOTAL_SAMPLES:-1000}" \
    $FUNC_ROUND_ARG

# Chain to next stage based on state
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

