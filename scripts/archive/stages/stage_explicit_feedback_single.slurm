#!/bin/bash
#SBATCH --job-name=explicit_fb_%x
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_explicit_feedback_%x_%j.out
#SBATCH --error=scripts/log/stage_explicit_feedback_%x_%j.err
#SBATCH --time=6:00:00
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

python src/pipeline/stages/stage_explicit_feedback_single.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "$SPEC_FILE" \
    --function_name "$FUNCTION_NAME" \
    --model_type "${MODEL_TYPE:-huggingface}" \
    --dsl_round "${DSL_ROUND:-0}" \
    $FUNC_ROUND_ARG \
    --num_iterations "${NUM_EXPLICIT_FEEDBACK_ITERATIONS:-30}"

# Chain to next stage based on state
PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -eq 0 ] || [ $PYTHON_EXIT_CODE -eq 1 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

