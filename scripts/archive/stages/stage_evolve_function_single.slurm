#!/bin/bash
#SBATCH --job-name=evolve_func_%x
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_evolve_function_%x_%j.out
#SBATCH --error=scripts/log/stage_evolve_function_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run the stage script
python src/pipeline/stages/stage_evolve_function_single.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --spec_file "$SPEC_FILE" \
    --function_name "$FUNCTION_NAME" \
    --failing_tasks $FAILING_TASKS \
    --model_type "${MODEL_TYPE:-huggingface}" \
    --dsl_round "${DSL_ROUND:-0}" \
    --func_evolution_round "${FUNC_EVOLUTION_ROUND:-0}" \
    --total_samples "${TOTAL_SAMPLES:-1000}"

# Chain to next stage based on state
if [ $? -eq 0 ] || [ $? -eq 1 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

