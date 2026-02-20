#!/bin/bash
#SBATCH --job-name=implement_cfg_%x
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_implement_cfg_%x_%j.out
#SBATCH --error=scripts/log/stage_implement_cfg_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Set PyTorch CUDA allocation config to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Note: TOTAL_SAMPLES environment variable controls the number of FunSearch samples
# Set it before submitting: export TOTAL_SAMPLES=1000
# Default is 1000 if not set

# Run the stage script
if [ -n "${FUNC_EVOLUTION_ROUND:-}" ]; then
    FUNC_ROUND_ARG="--func_evolution_round $FUNC_EVOLUTION_ROUND"
else
    FUNC_ROUND_ARG=""
fi

python src/pipeline/stages/stage_implement_cfg_single.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "${SPEC_FILE:-prompt_specifications/specification_with_updated_nld.txt}" \
    --function_name "$FUNCTION_NAME" \
    --model_type "${MODEL_TYPE:-huggingface}" \
    --dsl_round "${DSL_ROUND:-0}" \
    --total_samples "${TOTAL_SAMPLES:-1000}" \
    --num_iterations "${NUM_EXPLICIT_FEEDBACK_ITERATIONS:-30}" \
    $FUNC_ROUND_ARG

# Chain to next stage based on state (can be disabled)
if [ "${SKIP_CHAINING:-0}" != "1" ]; then
    if [ $? -eq 0 ] || [ $? -eq 1 ]; then
        source scripts/stages/chain_next_stage.sh
        chain_based_on_state
    fi
fi

