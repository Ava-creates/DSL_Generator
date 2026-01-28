#!/bin/bash
#SBATCH --job-name=generate_plots
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avani4@ualberta.ca
#SBATCH --output=scripts/log/stage_generate_plots_%j.out
#SBATCH --error=scripts/log/stage_generate_plots_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=aip-lelis

# Activate virtual environment
source new_dsl_env/bin/activate

# Set working directory to project root
cd /home/avani/projects/aip-lelis/avani/DSL_Generator

# Run plot generation
python -u generate_plots_from_metrics.py "$EXPERIMENT_DIR" --dsl_round "${DSL_ROUND:-0}"

