#!/bin/bash
#SBATCH --job-name=stage_get_cfg
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_get_cfg_%j.out
#SBATCH --error=scripts/log/stage_get_cfg_%j.err
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Load configuration from YAML file (if EXPERIMENT_CONFIG is set)
# Environment variables take precedence over config file
if [ -n "${EXPERIMENT_CONFIG:-}" ] || [ -f "config/experiment_config.yaml" ]; then
    source scripts/load_config.sh
fi

# Run the stage script
if [ "${SKIP_CFG_GENERATION:-false}" = "true" ]; then
    SKIP_FLAG="--skip_cfg_generation"
else
    SKIP_FLAG=""
fi

# Run the stage script
python src/pipeline/stages/stage_get_cfg.py \
    ${EXPERIMENT_DIR:+--experiment_dir "$EXPERIMENT_DIR"} \
    $SKIP_FLAG \
    ${CFG_OUTPUT_FILE:+--cfg_output_file "$CFG_OUTPUT_FILE"} \
    --max_cfg_retries "${MAX_CFG_RETRIES:-10}"

# If EXPERIMENT_DIR was auto-generated, read it from the file Python wrote
if [ -z "${EXPERIMENT_DIR:-}" ] && [ -f ".experiment_dir" ]; then
    export EXPERIMENT_DIR=$(cat .experiment_dir)
    echo "Using auto-generated experiment directory: $EXPERIMENT_DIR"
fi

# Chain to next stage based on state
if [ $? -eq 0 ]; then
    source scripts/stages/chain_next_stage.sh
    chain_based_on_state
fi

