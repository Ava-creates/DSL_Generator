#!/bin/bash
#SBATCH --job-name=funsearch_%FUNC_NAME%
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/funsearch_%FUNC_NAME%_%j.out
#SBATCH --error=scripts/log/funsearch_%FUNC_NAME%_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4  # 4 GPUs required for 1 vLLM instance (tensor_parallel_size=4)
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run FunSearch for the function
python3 scripts/run_funsearch_single_function.py \
    --func_name "$FUNC_NAME" \
    --func_file "$FUNC_FILE" \
    --func_init "$FUNC_INIT" \
    --spec_file "$SPEC_FILE" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --cfg_file "$CFG_FILE" \
    --desc_file "$DESC_FILE" \
    --model_type "$MODEL_TYPE" \
    --samples_per_prompt "$SAMPLES_PER_PROMPT" \
    --total_samples "$TOTAL_SAMPLES"

