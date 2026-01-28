#!/bin/bash
#SBATCH --job-name=dsl_cfg%j
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/cfg_version_%j.out
#SBATCH --error=scripts/log/cfg_version_%j.err
#SBATCH --time=2:00:00  # 2 days per CFG version
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4  # 4 GPUs required for 1 vLLM instance (tensor_parallel_size=4)
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run the CFG version wrapper script
bash scripts/run_cfg_version_wrapper.sh

