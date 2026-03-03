#!/bin/bash
#SBATCH --job-name=stage_explicit_fb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_explicit_feedback_%j.out
#SBATCH --error=scripts/log/stage_explicit_feedback_%j.err
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
log_chain_context "stage_explicit_feedback"

# Run the stage script
if [ -n "${FUNC_EVOLUTION_ROUND:-}" ]; then
    FUNC_ROUND_ARG="--func_evolution_round $FUNC_EVOLUTION_ROUND"
else
    FUNC_ROUND_ARG=""
fi

python src/pipeline/stages/stage_explicit_feedback.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "$SPEC_FILE" \
    --model_type "${MODEL_TYPE:-huggingface}" \
    --dsl_round "${DSL_ROUND:-0}" \
    $FUNC_ROUND_ARG \
    --num_iterations "${NUM_EXPLICIT_FEEDBACK_ITERATIONS:-1}"

