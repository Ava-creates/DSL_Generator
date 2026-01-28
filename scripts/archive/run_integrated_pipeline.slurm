#!/bin/bash
#SBATCH --job-name=unified_pipeline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/unified_pipeline_%j.out
#SBATCH --error=scripts/log/unified_pipeline_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4  # 4 GPUs required for 1 vLLM instance (tensor_parallel_size=4)
                      # All parallel funsearch workers share the same vLLM instance, so only 4 GPUs total needed
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run the wrapper script which handles checkpoint resumption
bash scripts/run_pipeline_wrapper.sh

