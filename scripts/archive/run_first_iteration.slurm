#!/bin/bash
#SBATCH --job-name=dsl_first_iter
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/first_iteration_%j.out
#SBATCH --error=scripts/log/first_iteration_%j.err
#SBATCH --time=60:00:00 
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4  # 4 GPUs required for 1 vLLM instance (tensor_parallel_size=4)
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run the first iteration wrapper script
bash scripts/run_first_iteration_wrapper.sh

